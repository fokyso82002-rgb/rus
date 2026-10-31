"""Run the task and measure it: base Soup and your solution, one after the other in this session.

    python harness/run_task.py                    # base, then solution -> runs/result.json
    python harness/run_task.py --only solution --out runs_try   # only your solution, quicker
    python harness/run_task.py --repeats 2        # base, solution, solution, base (less noise)
    python harness/run_task.py --profile          # plus a profiler table of every CUDA kernel

For each side it measures the wall time of the whole `soup train` command (start to saved
adapter, timed from outside; Triton compilation counts, every run starts with empty compiler
caches) and the peak GPU memory (nvidia-smi from outside, all GPUs), then scores the adapter
with eval_adapter.py. Your kernels' share of the training time is taken inside the same runs
(see _train.py). The jury runs this same script with its hidden rows: --train-file, --eval-file,
--seed, --repeats 2.
"""

import argparse
import json
import shutil
import statistics
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (MODEL, OFFLINE_ENV, REPO, installed_packages, library_versions,  # noqa: E402
                    read_json, run_logged, write_json)

LOG_KEYS = ("loss", "num_tokens", "entropy", "mean_token_accuracy", "grad_norm")


def load_task():
    import yaml

    return yaml.safe_load((REPO / "task" / "task.yaml").read_text(encoding="utf-8"))


def warm_up(out):
    """Download the model and read the Python packages once, so the first timed run does not
    pay for it."""
    code = ("import soup_cli.cli, transformers, peft, trl, bitsandbytes, triton; "
            "from huggingface_hub import snapshot_download; "
            f"print(snapshot_download({MODEL!r}))")
    exit_code, wall, _peak, _start = run_logged([sys.executable, "-c", code], out / "warmup.log",
                                                1800)
    if exit_code != 0:
        sys.exit(f"warm-up failed, see {out / 'warmup.log'}")
    print(f"[run_task] warm-up done in {wall} s", flush=True)


def make_config(run_dir, train_file, seed):
    import yaml

    config = load_task()
    config["data"]["train"] = str(train_file)
    config["training"]["seed"] = seed
    config["output"] = str(run_dir / "adapter")
    path = run_dir / "task.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


def adapter_tensors(adapter_dir):
    """Names, shapes and dtypes of the adapter tensors, read from the safetensors header."""
    path = Path(adapter_dir) / "adapter_model.safetensors"
    try:
        with open(path, "rb") as handle:
            size = struct.unpack("<Q", handle.read(8))[0]
            header = json.loads(handle.read(size))
    except (OSError, ValueError, struct.error):
        return None
    return {name: [info.get("dtype"), info.get("shape")] for name, info in sorted(header.items())
            if name != "__metadata__"}


def soup_outputs(adapter_dir):
    """What Soup logged and wrote: per-step metrics from the last checkpoint and the rewind log.
    A solution must keep these, not only the adapter."""
    states = sorted(Path(adapter_dir).glob("checkpoint-*/trainer_state.json"),
                    key=lambda path: int(path.parent.name.split("-")[-1]))
    state = read_json(states[-1], {}) if states else {}
    logged = [entry for entry in state.get("log_history", []) if "loss" in entry]
    rewind = []
    path = Path(adapter_dir) / "rewind.jsonl"
    if path.exists():
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                entry = json.loads(line)
                if entry.get("kind") == "batch":
                    rewind.append([entry.get("rows"), entry.get("row_tokens"), entry.get("row_loss")])
    return {"global_step": state.get("global_step"),
            "logged_metrics": sorted(logged[0]) if logged else [],
            "log": {key: [entry.get(key) for entry in logged] for key in LOG_KEYS},
            "rewind": rewind if path.exists() else None}


def train(side, name, args, profile=False):
    run_dir = args.out / name
    shutil.rmtree(run_dir, ignore_errors=True)
    run_dir.mkdir(parents=True)
    config = make_config(run_dir, args.train_file, args.seed)
    cmd = [sys.executable, str(REPO / "harness" / "_train.py"), "--config", str(config),
           "--metrics", str(run_dir / "metrics.json")]
    if side == "solution":
        cmd.append("--solution")
    if profile:
        cmd += ["--profile", str(run_dir / "profile.json")]
    print(f"[run_task] {name}: training ...", flush=True)
    env = dict(OFFLINE_ENV, SOUP_DB_PATH=str(run_dir / "experiments.db"),
               TRITON_CACHE_DIR=str(run_dir / "triton_cache"),
               TORCHINDUCTOR_CACHE_DIR=str(run_dir / "inductor_cache"))
    exit_code, wall, peak, start = run_logged(cmd, run_dir / "train.log", args.timeout, env)
    metrics = read_json(run_dir / "metrics.json", {})
    metrics.pop("step_s", None)
    result = {"run": name, "side": side, "exit_code": exit_code, "wall_s": wall,
              "gpu_peak_mib": peak, "gpu_start_mib": start, "metrics": metrics,
              **soup_outputs(run_dir / "adapter"),
              "adapter": str(run_dir / "adapter"),
              "adapter_config": read_json(run_dir / "adapter" / "adapter_config.json"),
              "adapter_tensors": adapter_tensors(run_dir / "adapter")}
    status = "ok" if exit_code == 0 else f"FAILED (exit {exit_code}, see {run_dir / 'train.log'})"
    print(f"[run_task] {name}: {status}, {wall} s, peak {peak} MiB", flush=True)
    return result


def evaluate(adapter, name, args):
    out_file = args.out / f"eval_{name}.json"
    out_file.unlink(missing_ok=True)
    exit_code, _wall, _peak, _start = run_logged(
        [sys.executable, str(REPO / "harness" / "eval_adapter.py"), "--adapter", str(adapter),
         "--eval-file", str(args.eval_file), "--out", str(out_file)],
        args.out / f"eval_{name}.log", args.timeout, OFFLINE_ENV)
    result = read_json(out_file, {}) if exit_code == 0 else {}
    return {"exit_code": exit_code, "mean_loss": result.get("mean_loss"), **result}


def summarize(runs, evals):
    sides = {}
    for side, side_runs in runs.items():
        if not side_runs or any(run["exit_code"] != 0 for run in side_runs):
            sides[side] = {"ok": False, "runs": len(side_runs),
                           "failed": [run["run"] for run in side_runs if run["exit_code"] != 0]}
            continue
        loops = [run["metrics"].get("train_loop_s") for run in side_runs
                 if run["metrics"].get("train_loop_s")]
        sides[side] = {"ok": True, "runs": len(side_runs),
                       "wall_s": round(statistics.median(run["wall_s"] for run in side_runs), 2),
                       "gpu_peak_mib": max(run["gpu_peak_mib"] for run in side_runs),
                       "train_loop_s": round(statistics.median(loops), 2) if loops else None,
                       "eval_loss": evals.get(side, {}).get("mean_loss")}
    base, sol = sides.get("base", {}), sides.get("solution", {})
    if base.get("ok") and sol.get("ok") and base["gpu_peak_mib"] and sol["gpu_peak_mib"]:
        ratios = {"speedup": round(base["wall_s"] / sol["wall_s"], 4),
                  "memory_ratio": round(base["gpu_peak_mib"] / sol["gpu_peak_mib"], 4)}
        if base["eval_loss"] and sol["eval_loss"]:
            ratios["quality_delta_pct"] = round(
                100 * (sol["eval_loss"] - base["eval_loss"]) / base["eval_loss"], 3)
        sides["ratios"] = ratios
    return sides


def main():
    task = load_task()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", choices=("base", "solution"))
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--profile", action="store_true",
                        help="also run your solution once under torch.profiler (for your analysis)")
    parser.add_argument("--train-file", default=str(REPO / "data" / "train.jsonl"))
    parser.add_argument("--eval-file", default=str(REPO / "data" / "eval_public.jsonl"))
    parser.add_argument("--seed", type=int, default=task["training"]["seed"])
    parser.add_argument("--out", default=str(REPO / "runs"))
    parser.add_argument("--timeout", type=int, default=900, help="seconds per run")
    args = parser.parse_args()
    args.out = Path(args.out).resolve()
    args.train_file = Path(args.train_file).resolve()
    args.eval_file = Path(args.eval_file).resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    for path in (args.train_file, args.eval_file):
        if not path.exists():
            sys.exit(f"{path} is missing: run `python task/prepare_data.py` first")
    for stale in ["result.json", "checks.json", *[p.name for p in args.out.glob("eval_*.json")]]:
        (args.out / stale).unlink(missing_ok=True)

    packages_before = installed_packages()
    warm_up(args.out)
    runs = {"base": [], "solution": []}
    for repeat in range(args.repeats):
        order = ("base", "solution") if repeat % 2 == 0 else ("solution", "base")
        for side in order:
            if args.only in (None, side):
                runs[side].append(train(side, f"{side}_{len(runs[side]) + 1}", args))
    runs = {side: side_runs for side, side_runs in runs.items() if side_runs}

    evals = {}
    for side, side_runs in runs.items():
        good = [run for run in side_runs if run["exit_code"] == 0]
        if good:
            print(f"[run_task] eval {side} ...", flush=True)
            evals[side] = evaluate(good[0]["adapter"], side, args)

    profile = None
    if args.profile and any(run["exit_code"] == 0 for run in runs.get("solution", [])):
        profile = train("solution", "solution_profile", args, profile=True)
    packages_after = installed_packages()

    result = {"task": {"train_file": str(args.train_file), "eval_file": str(args.eval_file),
                       "seed": args.seed, "repeats": args.repeats, "only": args.only},
              "versions": library_versions(),
              "packages_changed": sorted(set(packages_before) ^ set(packages_after)),
              "summary": summarize(runs, evals), "runs": runs, "evals": evals,
              "profile_run": profile}
    write_json(args.out / "result.json", result)

    print("\n=== result ===")
    for side in ("base", "solution"):
        info = result["summary"].get(side)
        if info and info.get("ok"):
            print(f"{side:9s} time {info['wall_s']:8.1f} s   peak {info['gpu_peak_mib']:6d} MiB"
                  f"   eval loss {info['eval_loss']}")
        elif info:
            print(f"{side:9s} FAILED: {', '.join(info['failed'])}")
    ratios = result["summary"].get("ratios")
    if ratios:
        print(f"speedup {ratios['speedup']:.3f}x   memory base/solution "
              f"{ratios['memory_ratio']:.3f}x   eval loss vs base "
              f"{ratios.get('quality_delta_pct')} %")
    if profile:
        print(f"profiler table: {args.out / 'solution_profile' / 'profile.json'}")
    print(f"written: {args.out / 'result.json'}")


if __name__ == "__main__":
    main()
