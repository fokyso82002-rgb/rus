"""One `soup train` run in this process, with or without your solution. Started by run_task.py.

What happens, in this order:
1. Soup's command line is set, and the CUDA context starts on GPU 0 with the harness allocator
   settings, before any code of yours: a run cannot move to another GPU or switch the allocator.
2. Every launch of a @triton.jit kernel is counted by name. From step 3 on, the launches of the
   kernels defined in files under solution/ are timed with CUDA events: this is the share of your
   kernels that check_rules.py reads, taken from the measured runs themselves.
3. `solution.apply()` runs (with --solution).
4. `soup train` runs; a trainer callback records the step times and the trainer settings.
5. metrics.json is written.
With --profile the run also writes a torch.profiler table of every CUDA kernel for a few steps
and stops early; that is for your own analysis, the checks do not use it.
The jury uses this file as it is; do not edit it.
"""

import argparse
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ALLOC_CONF, RUNTIME_BANNED, SOLUTION_DIR, write_json  # noqa: E402

TIMED_FROM_STEP = 2  # 0-based; the first steps compile and autotune
PROFILE_WAIT, PROFILE_WARMUP, PROFILE_ACTIVE = 6, 2, 16
TRAINER_ARGS = ("fp16", "bf16", "per_device_train_batch_size", "gradient_accumulation_steps",
                "learning_rate", "num_train_epochs", "max_steps", "seed", "data_seed", "optim",
                "lr_scheduler_type", "warmup_steps", "warmup_ratio", "weight_decay",
                "max_grad_norm", "gradient_checkpointing", "torch_compile", "logging_steps",
                "max_length", "packing", "padding_free", "completion_only_loss",
                "assistant_only_loss", "loss_type", "use_liger_kernel", "activation_offloading")


def plain(value):
    """Trainer settings as JSON values (optim and scheduler are enums)."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(getattr(value, "value", value))


class TritonWatch:
    """Counts the launches of every @triton.jit kernel by name and times, with CUDA events, the
    launches of the kernels defined in files under solution/ while `timing` is on."""

    def __init__(self, torch):
        self.torch, self.launches, self.files, self.events = torch, {}, {}, []
        self.timing, self.error = False, None
        try:
            from triton.runtime.jit import JITFunction
        except Exception as exc:  # noqa: BLE001 - without Triton there is nothing to watch
            self.error = repr(exc)
            return
        original_run = JITFunction.run
        solution_dir = str(SOLUTION_DIR.resolve()) + os.sep
        watch = self

        def run(jit, *args, **kwargs):
            fn = getattr(jit, "fn", None)
            name = getattr(fn, "__name__", "?")
            watch.launches[name] = watch.launches.get(name, 0) + 1
            path = os.path.abspath(getattr(getattr(fn, "__code__", None), "co_filename", "") or "")
            watch.files.setdefault(name, path)
            if not watch.timing or kwargs.get("warmup") or not path.startswith(solution_dir):
                return original_run(jit, *args, **kwargs)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            result = original_run(jit, *args, **kwargs)
            end.record()
            watch.events.append((name, start, end))
            return result

        JITFunction.run = run

    def own_ms(self):
        self.torch.cuda.synchronize()
        totals = {}
        for name, start, end in self.events:
            totals[name] = totals.get(name, 0.0) + start.elapsed_time(end)
        return {name: round(ms, 3) for name, ms in totals.items()}


def kernel_table(prof):
    from torch.autograd import DeviceType

    rows = []
    for event in prof.key_averages():
        if event.device_type != DeviceType.CUDA:
            continue
        # Ranges such as "ProfilerStep#8" are copied onto the GPU timeline and span every
        # kernel of the step; they are labels, not kernels.
        if getattr(event, "is_user_annotation", False) or event.key.startswith("ProfilerStep"):
            continue
        time_us = getattr(event, "self_device_time_total", None)
        if time_us is None:
            time_us = getattr(event, "self_cuda_time_total", 0)
        rows.append({"name": event.key, "time_us": float(time_us), "count": int(event.count)})
    rows.sort(key=lambda row: -row["time_us"])
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--solution", action="store_true")
    parser.add_argument("--profile", help="write a per-kernel GPU time table here and stop early")
    args = parser.parse_args()
    sys.argv = ["soup", "train", "--config", args.config, "--yes"]

    # Soup imports unsloth by itself when it is installed (for a tip), which would patch the
    # model behind both runs; keep it out.
    sys.modules["unsloth"] = None
    sys.modules["unsloth_zoo"] = None
    import torch

    torch.empty(1, device="cuda")
    import transformers

    watch = TritonWatch(torch)
    apply_s = None
    if args.solution:
        sys.path.insert(0, str(REPO))
        applying = time.perf_counter()
        import solution

        solution.apply()
        apply_s = round(time.perf_counter() - applying, 2)

    marks = {"first_step_begin": None, "last_step_end": None, "step_s": [], "trainer": None}
    prof = None
    if args.profile:
        from torch.profiler import ProfilerActivity, profile, schedule

        prof = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                       schedule=schedule(wait=PROFILE_WAIT, warmup=PROFILE_WARMUP,
                                         active=PROFILE_ACTIVE, repeat=1))
        prof.start()

    class Probe(transformers.TrainerCallback):
        def on_step_begin(self, train_args, state, control, **kwargs):
            torch.cuda.synchronize()
            self.step_start = time.perf_counter()
            if marks["first_step_begin"] is None:
                marks["first_step_begin"] = self.step_start
            watch.timing = len(marks["step_s"]) >= TIMED_FROM_STEP

        def on_step_end(self, train_args, state, control, **kwargs):
            torch.cuda.synchronize()
            watch.timing = False
            now = time.perf_counter()
            marks["last_step_end"] = now
            marks["step_s"].append(round(now - self.step_start, 4))
            if prof is not None:
                prof.step()
                if len(marks["step_s"]) >= PROFILE_WAIT + PROFILE_WARMUP + PROFILE_ACTIVE:
                    control.should_training_stop = True

    original_init = transformers.Trainer.__init__

    def init(self, *init_args, **init_kwargs):
        original_init(self, *init_args, **init_kwargs)
        self.add_callback(Probe())
        marks["trainer"] = {"class": type(self).__name__,
                            **{name: plain(getattr(self.args, name, None))
                               for name in TRAINER_ARGS}}

    transformers.Trainer.__init__ = init

    code = 0
    try:
        from soup_cli.cli import run

        run()
    except SystemExit as exc:
        code = 0 if exc.code is None else (exc.code if isinstance(exc.code, int) else 1)

    steps = marks["step_s"]
    metrics = {
        "exit_code": code,
        "solution": args.solution,
        "apply_s": apply_s,
        "steps": len(steps),
        "train_loop_s": (round(marks["last_step_end"] - marks["first_step_begin"], 2)
                         if steps else None),
        "median_step_s": sorted(steps)[len(steps) // 2] if steps else None,
        "step_s": steps,
        "timed_steps": max(0, len(steps) - TIMED_FROM_STEP),
        "timed_loop_s": round(sum(steps[TIMED_FROM_STEP:]), 3),
        "own_kernel_ms": watch.own_ms(),
        "triton_launches": dict(watch.launches),
        "triton_files": {name: os.path.relpath(path, REPO) if path.startswith(str(REPO)) else path
                         for name, path in watch.files.items()},
        "triton_watch_error": watch.error,
        "torch_max_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1),
        "torch_max_reserved_mib": round(torch.cuda.max_memory_reserved() / 2**20, 1),
        "cuda_device": torch.cuda.get_device_name(0),
        "device_count": torch.cuda.device_count(),
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
        "alloc_conf_expected": ALLOC_CONF,
        "trainer": marks["trainer"],
        "banned_modules_loaded": sorted({name.split(".")[0]
                                         for name, module in list(sys.modules.items())
                                         if module is not None
                                         and name.split(".")[0] in RUNTIME_BANNED}),
    }
    if prof is not None:
        prof.stop()
        table = kernel_table(prof)
        write_json(args.profile, {"active_steps": PROFILE_ACTIVE,
                                  "total_gpu_us": sum(row["time_us"] for row in table),
                                  "kernels": table})
    write_json(args.metrics, metrics)
    sys.exit(code)


if __name__ == "__main__":
    main()
