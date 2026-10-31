"""Check the rules on a finished run (reads runs/result.json written by run_task.py).

    python harness/check_rules.py            # prints every check, writes runs/checks.json

Exit code 0 means every check passed. These checks are the automatic part of the rules; the
jury also reads the code in solution/, and its review has the last word.
"""

import argparse
import ast
import copy
import difflib
import hashlib
import json
import math
import os
import re
import statistics
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (ALLOC_CONF, BANNED_MODULES, ENV, EXAMPLE_PREFIX,  # noqa: E402
                    INDUCTOR_PREFIXES, REPO, SOLUTION_DIR, write_json)

MIN_KERNEL_SHARE = 0.05      # your kernels together: at least 5% of the training time
MAX_QUALITY_DELTA_PCT = 0.5  # eval loss at most 0.5% above the base (higher loss is worse)
MAX_FIRST_LOSS_DELTA = 0.01  # the first step trains from the same start: loss within 1%
MAX_CURVE_DELTA = 0.03       # mean relative gap over all steps (loss, entropy, accuracy, rewind)
MAX_SLOWDOWN = 1.10          # not slower than the base by more than 10% ...
MAX_MEMORY_GROWTH = 1.10     # ... and not more memory by more than 10%
MAX_IDLE_START_MIB = 1000    # the GPUs are idle when a run starts
EXAMPLE_COPY = 0.95          # a kernel this close to the example is the example: it does not count
EXAMPLE_REVIEW = 0.8         # this close: it counts, and the jury reads it by hand
VOLATILE_ADAPTER_KEYS = ("peft_version", "inference_mode", "revision")
CHECK_COUNT = 10
EXPECTED_TESTS = 79          # tests/ as copied from Soup v0.77.0


def triton_names(tree):
    """Names bound to the triton module and to triton.jit in one file."""
    modules, jits = set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "triton":
                    modules.add(alias.asname or "triton")
        elif isinstance(node, ast.ImportFrom) and node.module == "triton":
            for alias in node.names:
                if alias.name == "jit":
                    jits.add(alias.asname or "jit")
    return modules, jits


def is_triton_jit(decorator, modules, jits):
    """@triton.jit, @triton.jit(...), or the jit imported from triton under any name."""
    target = decorator.func if isinstance(decorator, ast.Call) else decorator
    if isinstance(target, ast.Attribute):
        return target.attr == "jit" and isinstance(target.value, ast.Name) and \
            target.value.id in modules
    return isinstance(target, ast.Name) and target.id in jits


def canonical(func):
    """The kernel's structure: no name, docstring or identifiers, keyword arguments sorted,
    operands of + and * in a fixed order. Renamed or reshuffled copies give the same text."""
    func = copy.deepcopy(func)
    func.name, func.decorator_list, func.returns = "kernel", [], None
    if func.body and isinstance(func.body[0], ast.Expr) and \
            isinstance(func.body[0].value, ast.Constant) and isinstance(func.body[0].value.value, str):
        func.body = func.body[1:]
    for node in ast.walk(func):
        if isinstance(node, ast.Name):
            node.id = "_"
        elif isinstance(node, ast.arg):
            node.arg, node.annotation = "_", None
        elif isinstance(node, ast.Call):
            node.keywords.sort(key=lambda keyword: keyword.arg or "")

    class Commute(ast.NodeTransformer):
        def visit_BinOp(self, node):
            self.generic_visit(node)
            if isinstance(node.op, (ast.Add, ast.Mult)) and \
                    ast.dump(node.right) < ast.dump(node.left):
                node.left, node.right = node.right, node.left
            return node

    return ast.dump(Commute().visit(func), annotate_fields=False)


def triton_kernels(path):
    tree = ast.parse(Path(path).read_text(encoding="utf-8"), filename=str(path))
    modules, jits = triton_names(tree)
    kernels = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
               and any(is_triton_jit(d, modules, jits) for d in node.decorator_list)]
    return tree, kernels


EXAMPLE_SHAPES = [canonical(node)
                  for node in triton_kernels(Path(__file__).parent / "example_reference.py")[1]]


def example_likeness(node):
    shape = canonical(node)
    return max(difflib.SequenceMatcher(None, shape, example).ratio() for example in EXAMPLE_SHAPES)


def source_kernels(solution_dir):
    """The @triton.jit functions in solution/ that may count as yours, the copies of the example,
    kernels close to the example (for the jury), imports of banned libraries, and files that do
    not parse."""
    own, copies, lookalikes, banned, broken = {}, [], [], [], []
    for path in sorted(Path(solution_dir).resolve().rglob("*.py")):
        where = str(path.relative_to(REPO) if path.is_relative_to(REPO) else path)
        try:
            tree, kernels = triton_kernels(path)
        except (SyntaxError, UnicodeDecodeError, ValueError) as exc:
            broken.append(f"{where}: {exc.__class__.__name__}")
            continue
        for node in kernels:
            if node.name.startswith(INDUCTOR_PREFIXES):
                continue
            likeness = example_likeness(node)
            if node.name.startswith(EXAMPLE_PREFIX) or likeness >= EXAMPLE_COPY:
                copies.append(node.name)
                continue
            own[node.name] = where
            if likeness >= EXAMPLE_REVIEW:
                lookalikes.append(f"{node.name} ({likeness:.2f})")
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            banned += [f"{where}: {name}" for name in names if name.split(".")[0] in BANNED_MODULES]
    return own, copies, lookalikes, banned, broken


def solution_sha256(solution_dir):
    digest = hashlib.sha256()
    for path in sorted(Path(solution_dir).rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts:
            digest.update(str(path.relative_to(solution_dir)).encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def relative(a, b):
    return abs(a - b) / max(abs(b), 1e-12)


def mean_relative(values, reference):
    """Mean relative gap of two series of finite numbers of the same length; None otherwise."""
    if not isinstance(values, list) or not isinstance(reference, list) or not values \
            or len(values) != len(reference) or not all(map(finite, values + reference)):
        return None
    return sum(relative(a, b) for a, b in zip(values, reference)) / len(values)


def as_dict(value):
    return value if isinstance(value, dict) else {}


def same_adapter_config(base, sol):
    if not isinstance(base, dict) or not isinstance(sol, dict) or not base or not sol:
        return ["adapter_config.json missing"]
    diffs = []
    for key in sorted(set(base) | set(sol)):
        if key in VOLATILE_ADAPTER_KEYS:
            continue
        left, right = base.get(key), sol.get(key)
        if isinstance(left, list) and isinstance(right, list):
            left, right = sorted(map(str, left)), sorted(map(str, right))
        if left != right:
            diffs.append(f"adapter {key}: {left} vs {right}")
    return diffs


def compare_outputs(base, sol):
    """Everything about the trained task and Soup's outputs that must match the base run.
    Anything missing on either side is a difference."""
    diffs = []
    for key in ("global_step", "logged_metrics", "adapter_tensors"):
        if not base.get(key) or not sol.get(key):
            diffs.append(f"{key} missing")
        elif base.get(key) != sol.get(key):
            diffs.append(f"{key} differs" if key == "adapter_tensors"
                         else f"{key}: {base.get(key)} vs {sol.get(key)}")
    diffs += same_adapter_config(base.get("adapter_config"), sol.get("adapter_config"))
    base_args = as_dict(as_dict(base.get("metrics")).get("trainer"))
    sol_args = as_dict(as_dict(sol.get("metrics")).get("trainer"))
    if not base_args or not sol_args:
        diffs.append("trainer settings missing")
    for key in sorted(set(base_args) | set(sol_args)):
        if key != "class" and base_args.get(key) != sol_args.get(key):
            diffs.append(f"trainer {key}: {base_args.get(key)} vs {sol_args.get(key)}")
    base_log, sol_log = as_dict(base.get("log")), as_dict(sol.get("log"))
    tokens = base_log.get("num_tokens")
    if not isinstance(tokens, list) or not tokens or not all(map(finite, tokens)) \
            or tokens != sol_log.get("num_tokens"):
        diffs.append("tokens per step missing or different (the batches are not the same)")
    for key in ("entropy", "mean_token_accuracy"):
        gap = mean_relative(sol_log.get(key), base_log.get(key))
        if gap is None or gap > MAX_CURVE_DELTA:
            diffs.append(f"{key}: " + ("missing or not comparable" if gap is None
                                       else f"differs from the base by {100 * gap:.2f}%"))
    base_rw, sol_rw = base.get("rewind"), sol.get("rewind")
    if not isinstance(base_rw, list) or not isinstance(sol_rw, list) or not base_rw \
            or len(base_rw) != len(sol_rw) \
            or not all(isinstance(x, list) and len(x) == 3 for x in base_rw + sol_rw):
        diffs.append("rewind.jsonl missing, malformed or with another number of batches")
    else:
        if any(b[0] != s[0] or b[1] != s[1] for b, s in zip(base_rw, sol_rw)):
            diffs.append("rewind.jsonl: rows or token counts differ (batches changed)")
        gap = mean_relative([x for s in sol_rw for x in (s[2] or [])],
                            [x for b in base_rw for x in (b[2] or [])])
        if gap is None or gap > MAX_CURVE_DELTA:
            diffs.append("rewind.jsonl row losses " + ("missing or not comparable" if gap is None
                                                       else f"differ by {100 * gap:.2f}%"))
    return diffs


def compare_curves(base, run):
    """(first-step gap, mean gap over all steps) of the loss curve; None where not comparable."""
    b_loss, s_loss = as_dict(base.get("log")).get("loss"), as_dict(run.get("log")).get("loss")
    if not isinstance(b_loss, list) or not isinstance(s_loss, list) or not b_loss or not s_loss \
            or not finite(b_loss[0]) or not finite(s_loss[0]):
        return None, None
    return relative(s_loss[0], b_loss[0]), mean_relative(s_loss, b_loss)


def run_tests():
    env = dict(ENV, SOUP_APPLY_SOLUTION="1")
    try:
        proc = subprocess.run([sys.executable, "-m", "pytest", "tests", "-q", "-p",
                               "no:cacheprovider"], cwd=REPO, env=env, capture_output=True,
                              text=True, timeout=1800, check=False)
    except subprocess.TimeoutExpired:
        return False, "timed out after 1800 s"
    lines = [line for line in (proc.stdout + "\n" + proc.stderr).splitlines() if line.strip()]
    summary = [line for line in lines if " passed" in line or " failed" in line or "error" in line]
    last = (summary or lines or ["no output"])[-1]
    exact = re.search(rf"(?<!\d){EXPECTED_TESTS} passed\b", last) is not None
    return proc.returncode == 0 and exact and " failed" not in last, last


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--result", default=str(REPO / "runs" / "result.json"))
    parser.add_argument("--skip-tests", action="store_true")
    args = parser.parse_args()
    result_path = Path(args.result)
    checks_path = result_path.parent / "checks.json"
    checks_path.unlink(missing_ok=True)
    try:
        raw = result_path.read_bytes()
        result = json.loads(raw)
    except (OSError, ValueError) as exc:
        sys.exit(f"cannot read {result_path} ({exc.__class__.__name__}): "
                 "run `python harness/run_task.py` first")
    if not isinstance(result, dict) or not isinstance(result.get("runs"), dict):
        sys.exit(f"{result_path} is not a result of harness/run_task.py")
    result_sha = hashlib.sha256(raw).hexdigest()  # the bytes these checks read, nothing later
    solution_sha = solution_sha256(SOLUTION_DIR)

    checks, notes = [], []

    def check(name, fn):
        try:
            ok, detail = fn()
        except Exception as exc:  # noqa: BLE001 - a malformed result fails the check, no crash
            ok, detail = False, f"malformed result.json ({exc.__class__.__name__}: {exc})"
        checks.append({"check": name, "ok": ok is True, "detail": detail})

    def runs_of(side):
        value = result["runs"].get(side, [])
        return [run for run in value if isinstance(run, dict)] if isinstance(value, list) else []

    def metrics(run):
        return as_dict(run.get("metrics"))

    repeats = as_dict(result.get("task")).get("repeats", 1)
    base_runs, sol_runs = runs_of("base"), runs_of("solution")
    good_base = [run for run in base_runs if run.get("exit_code") == 0]
    good_sol = [run for run in sol_runs if run.get("exit_code") == 0]
    all_good = bool(good_base) and bool(good_sol) and len(good_base) == len(base_runs) \
        and len(good_sol) == len(sol_runs)
    own, copies, lookalikes, banned_imports, broken = source_kernels(SOLUTION_DIR)

    def c1():
        failed = [str(run.get("run")) for run in base_runs + sol_runs if run.get("exit_code") != 0]
        ok = isinstance(repeats, int) and repeats >= 1 and len(base_runs) == repeats \
            and len(sol_runs) == repeats and not failed
        return ok, (f"{len(base_runs)} base and {len(sol_runs)} solution runs of {repeats} each"
                    + (f"; failed: {', '.join(failed)}" if failed else ""))

    def c2():
        loaded, unrecorded = set(), []
        for run in sol_runs:
            names = metrics(run).get("banned_modules_loaded")
            if isinstance(names, list):
                loaded |= {str(name) for name in names}
            else:
                unrecorded.append(str(run.get("run")))
        changed = result.get("packages_changed")
        changed = changed if isinstance(changed, list) else ["not recorded"]
        ok = bool(sol_runs) and not loaded and not unrecorded and not banned_imports and not changed
        return ok, (f"loaded: {sorted(loaded) or 'none'}; imported in solution/: "
                    f"{banned_imports or 'none'}; packages changed during the run: "
                    f"{changed or 'none'}"
                    + (f"; not recorded in {', '.join(unrecorded)}" if unrecorded else ""))

    def launched_in(run):
        """Your kernels that this run launched from a file under solution/."""
        files = as_dict(metrics(run).get("triton_files"))
        launches = as_dict(metrics(run).get("triton_launches"))
        found = set()
        for name, count in launches.items():
            where = os.path.normpath(str(files.get(name, ""))).replace("\\", "/")
            if finite(count) and count > 0 and name in own and where.startswith("solution/"):
                found.add(name)
        return found

    per_run = [launched_in(run) for run in good_sol]
    launched = set.intersection(*per_run) if per_run else set()

    def c3():
        not_run = sorted(set(own) - launched)
        detail = ", ".join(f"{n} ({own[n]})" for n in sorted(launched)) or "none ran in every run"
        if not_run:
            detail += f"; not launched in every solution run: {', '.join(not_run)}"
        if copies:
            detail += f"; the example's kernels do not count, under any name: {', '.join(copies)}"
        if broken:
            detail += f"; files that do not parse: {', '.join(broken)}"
        return bool(launched) and not broken and bool(good_sol) and \
            len(good_sol) == len(sol_runs), detail

    def c4():
        shares = []
        for run in good_sol:
            kernel_ms = as_dict(metrics(run).get("own_kernel_ms"))
            own_ms = sum(ms for name, ms in kernel_ms.items() if name in launched and finite(ms))
            loop_s = metrics(run).get("timed_loop_s")
            shares.append(own_ms / (1000 * loop_s) if finite(loop_s) and loop_s > 0 else 0.0)
        if not shares:
            return False, "no solution run"
        share = min(shares)
        if MIN_KERNEL_SHARE <= share < MIN_KERNEL_SHARE + 0.01:
            notes.append("your kernels are within 1 point of the 5% line; the jury's rows can "
                         "differ")
        return share >= MIN_KERNEL_SHARE, (f"{100 * share:.1f}% of the training time (lowest of "
                                           f"{len(shares)} runs, steps 3+)")

    def c5():
        if not good_base or not good_sol:
            return False, "needs a finished base run and a finished solution run"
        diffs = []
        for run in good_base[1:] + good_sol:
            diffs += [f"{run.get('run')}: {d}" for d in compare_outputs(good_base[0], run)]
        return not diffs, "; ".join(diffs) or "identical within tolerance"

    def c6():
        if not good_base or not good_sol:
            return False, "needs a finished base run and a finished solution run"
        parts, ok = [], True
        for run in good_base[1:] + good_sol:
            first, mean = compare_curves(good_base[0], run)
            if first is None or mean is None:
                ok = False
                parts.append(f"{run.get('run')}: losses missing or not comparable")
                continue
            ok = ok and first <= MAX_FIRST_LOSS_DELTA and mean <= MAX_CURVE_DELTA
            parts.append(f"{run.get('run')}: first step {100 * first:.2f}%, mean of all steps "
                         f"{100 * mean:.2f}%")
        return ok, "; ".join(parts) + (f" (max {100 * MAX_FIRST_LOSS_DELTA:.0f}% and "
                                       f"{100 * MAX_CURVE_DELTA:.0f}%)")

    evals = as_dict(result.get("evals"))
    base_loss = as_dict(evals.get("base")).get("mean_loss")
    sol_loss = as_dict(evals.get("solution")).get("mean_loss")

    def c7():
        if not all_good or not finite(base_loss) or not finite(sol_loss) or base_loss <= 0:
            return False, "eval missing (needs finished base and solution runs)"
        delta = 100 * (sol_loss - base_loss) / base_loss
        if delta < -1.0:
            notes.append(f"eval loss is {-delta:.2f}% below the base; the jury will look at how "
                         "the adapter is produced")
        return delta <= MAX_QUALITY_DELTA_PCT, f"{delta:+.3f}%"

    def c8():
        if not all_good:
            return False, "needs finished base and solution runs"
        values = [run.get(key) for run in good_base + good_sol for key in ("wall_s", "gpu_peak_mib")]
        if not all(finite(v) and v > 0 for v in values):
            return False, "times or peaks missing"
        slower = statistics.median(r["wall_s"] for r in good_sol) / \
            statistics.median(r["wall_s"] for r in good_base)
        heavier = max(r["gpu_peak_mib"] for r in good_sol) / \
            max(r["gpu_peak_mib"] for r in good_base)
        ratios = as_dict(as_dict(result.get("summary")).get("ratios"))
        stated = (ratios.get("speedup"), ratios.get("memory_ratio"))
        if not all(finite(v) and v > 0 for v in stated) or \
                relative(1 / stated[0], slower) > 0.005 or relative(1 / stated[1], heavier) > 0.005:
            return False, "the summary in result.json does not match its runs"
        return slower <= MAX_SLOWDOWN and heavier <= MAX_MEMORY_GROWTH, \
            f"time x{slower:.3f}, memory x{heavier:.3f} of the base"

    def c9():
        problems = []
        for run in base_runs + sol_runs:
            name, m, start = run.get("run"), metrics(run), run.get("gpu_start_mib")
            if not finite(start) or start > MAX_IDLE_START_MIB:
                problems.append(f"{name}: GPUs not idle at start ({start} MiB)")
            if run.get("exit_code") != 0:
                continue
            peak, reserved = run.get("gpu_peak_mib"), m.get("torch_max_reserved_mib")
            if not finite(peak) or not finite(reserved) or peak < 0.97 * reserved:
                problems.append(f"{name}: nvidia-smi peak {peak} MiB, PyTorch's reserved peak "
                                f"{reserved} MiB")
            if "T4" not in str(m.get("cuda_device")):
                problems.append(f"{name}: not a Tesla T4 ({m.get('cuda_device')})")
            if m.get("visible_devices") != "0" or m.get("device_count") != 1 \
                    or m.get("alloc_conf") != ALLOC_CONF:
                problems.append(f"{name}: GPU or allocator settings changed "
                                f"({m.get('visible_devices')}, {m.get('device_count')} devices, "
                                f"{m.get('alloc_conf')})")
        return bool(base_runs + sol_runs) and not problems, "; ".join(problems) or "clean"

    check("1. every base and solution run finished", c1)
    check("2. no ready-made kernel libraries", c2)
    check("3. own @triton.jit kernels in solution/ run in training", c3)
    check("4. own kernels together take >= 5% of the training time", c4)
    check("5. same task and Soup outputs (steps, adapter, settings, tokens, metrics, rewind)", c5)
    check("6. loss curve follows the base", c6)
    check(f"7. quality: eval loss at most {MAX_QUALITY_DELTA_PCT}% above the base", c7)
    check("8. not worse than the base on time or memory (10% margin)", c8)
    check("9. clean measurement (a T4, idle GPUs, one GPU, harness allocator, peaks agree)", c9)
    if args.skip_tests:
        check("10. Soup regression tests with your patches",
              lambda: (False, "skipped (--skip-tests)"))
    else:
        check("10. Soup regression tests with your patches", run_tests)
    if lookalikes:
        notes.append(f"kernels close to the example, which the jury will read by hand: "
                     f"{', '.join(lookalikes)}; the example's SwiGLU does not count however it "
                     "is rewritten")

    write_json(checks_path, {"result_sha256": result_sha, "solution_sha256": solution_sha,
                             "checks": checks, "notes": notes})
    width = max(len(c["check"]) for c in checks)
    for c in checks:
        print(f"{'PASS' if c['ok'] else 'FAIL'}  {c['check']:<{width}}  {c['detail']}")
    for note in notes:
        print(f"NOTE  {note}")
    passed = len(checks) == CHECK_COUNT and all(c["ok"] for c in checks)
    print("\nALL CHECKS PASSED" if passed else "\nSOME CHECKS FAILED")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
