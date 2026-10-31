"""Shared helpers of the harness. The jury runs these same files; do not edit them."""

import hashlib
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SOLUTION_DIR = REPO / "solution"
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"  # the same `base` as task/task.yaml

# Ready-made kernel libraries. solution/ may not import any of them (checked in its source), and
# none may be loaded during a run, except torchao, which peft can import by itself.
BANNED_MODULES = (
    "unsloth", "unsloth_zoo", "liger_kernel", "xformers", "flash_attn", "flashinfer",
    "apex", "torchao", "bitblas", "gemlite", "hqq", "cut_cross_entropy", "deepspeed",
    "vllm", "awq", "auto_gptq", "gptqmodel", "exllamav2", "marlin", "quack", "tilelang",
)
RUNTIME_BANNED = tuple(name for name in BANNED_MODULES if name != "torchao")

# Prefixes of the kernels torch.compile (Inductor) generates; they never count as your own.
INDUCTOR_PREFIXES = ("triton_poi_", "triton_per_", "triton_red_", "triton_tem_", "triton_for_",
                     "triton_spl_", "triton_unk_")

# Kernels of the shipped example; they never count as your own, and neither do copies of them.
EXAMPLE_PREFIX = "example_"

ALLOC_CONF = "expandable_segments:True"

# GPU 0 only (Kaggle gives two; Soup would spread the model over both); the same allocator
# settings for base and solution (expandable segments keep the allocator from fragmenting on
# batches of varying length, so the measured peak follows the tensors that are alive); no
# Hugging Face network calls after the warm-up download; no audit log.
ENV = dict(os.environ, CUDA_VISIBLE_DEVICES="0", CUDA_DEVICE_ORDER="PCI_BUS_ID",
           PYTORCH_CUDA_ALLOC_CONF=ALLOC_CONF, PYTHONUNBUFFERED="1",
           TOKENIZERS_PARALLELISM="false", SOUP_NO_AUDIT_LOG="1")
OFFLINE_ENV = dict(ENV, HF_HUB_OFFLINE="1")

VERSION_PACKAGES = ("soup-cli", "torch", "triton", "transformers", "peft", "trl", "bitsandbytes",
                    "accelerate", "datasets")


class GpuPoller:
    """Peak GPU memory.used (MiB) summed over every GPU of the machine, sampled every 100 ms by
    nvidia-smi from outside the measured process. This is the memory number we score. `start` is
    the sum at the first sample, before the measured process allocates anything."""

    def __enter__(self):
        self.peak, self.start, self.current = 0, None, {}
        listing = subprocess.run(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                                 capture_output=True, text=True, check=False).stdout
        self.gpus = len(listing.split())
        self.proc = subprocess.Popen(
            ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits",
             "-lms", "100"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()
        return self

    def _read(self):
        for line in self.proc.stdout:
            try:
                index, used = (int(part) for part in line.split(","))
            except ValueError:
                continue
            self.current[index] = used
            if len(self.current) >= self.gpus:
                total = sum(self.current.values())
                if self.start is None:
                    self.start = total
                self.peak = max(self.peak, total)

    def __exit__(self, *exc):
        self.proc.terminate()
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        self.reader.join(timeout=2)
        return False


def kill_group(proc):
    """Kill the process and everything it started (a run must not leave work on the GPU)."""
    if hasattr(os, "killpg"):
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    elif proc.poll() is None:
        proc.kill()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass


def run_logged(cmd, log_path, timeout, env=None):
    """Run `cmd` in its own process group, write its output to `log_path`; return (exit code,
    wall seconds, peak MiB, GPU memory in use when it started in MiB). The clock stops when the
    process exits."""
    started = time.perf_counter()
    with GpuPoller() as poller, open(log_path, "w", encoding="utf-8") as log:
        proc = subprocess.Popen(cmd, cwd=REPO, env=env or ENV, stdout=log,
                                stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            code = -9
        wall = round(time.perf_counter() - started, 2)
        kill_group(proc)
        if code == -9:
            log.write(f"\nTIMEOUT after {timeout} s\n")
    return code, wall, poller.peak, poller.start


def installed_packages():
    """`pip freeze` lines; the harness compares them before and after the runs."""
    out = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True,
                         check=False).stdout
    return sorted(line for line in out.splitlines() if line.strip())


def library_versions():
    from importlib import metadata

    versions = {"python": sys.version.split()[0]}
    for name in VERSION_PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    gpus = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                          capture_output=True, text=True, check=False).stdout
    versions["gpus"] = [line.strip() for line in gpus.splitlines() if line.strip()]
    return versions


def file_sha256(path):
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


def read_json(path, default=None):
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return default


def write_json(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
