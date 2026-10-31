"""Fixtures the five regression tests need, taken from Soup v0.77.0 tests/conftest.py (Apache-2.0).

The tests run against the installed `soup-cli==0.77.0`. With SOUP_APPLY_SOLUTION=1 (what
`harness/check_rules.py` sets) your `solution.apply()` runs first, so the tests check that your
patches do not break Soup.
"""

import json
import os
import sys
from pathlib import Path

import pytest

if os.environ.get("SOUP_APPLY_SOLUTION") == "1":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import solution

    solution.apply()


@pytest.fixture(autouse=True)
def _isolate_experiments_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point Soup's experiments DB at a per-test temp file."""
    monkeypatch.setenv("SOUP_DB_PATH", str(tmp_path / "experiments.db"))


@pytest.fixture(autouse=True)
def _no_ambient_tool_auth_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SOUP_TOOL_AUTH_TOKEN", raising=False)


@pytest.fixture(autouse=True)
def _restore_working_directory():
    before = os.getcwd()
    yield
    os.chdir(before)


@pytest.fixture
def tmp_data_dir(tmp_path: Path) -> Path:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    return data_dir


@pytest.fixture
def sample_alpaca_data(tmp_data_dir: Path) -> Path:
    path = tmp_data_dir / "train.jsonl"
    samples = [
        {"instruction": "What is Python?", "input": "",
         "output": "Python is a programming language."},
        {"instruction": "Explain gravity", "input": "",
         "output": "Gravity is a fundamental force."},
        {"instruction": "Translate hello to Spanish", "input": "hello", "output": "hola"},
    ]
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
    return path


@pytest.fixture
def sample_config(tmp_path: Path, sample_alpaca_data: Path) -> Path:
    config_path = tmp_path / "soup.yaml"
    config_path.write_text(
        f"""base: meta-llama/Llama-3.1-8B-Instruct
task: sft
data:
  train: {sample_alpaca_data}
  format: alpaca
  val_split: 0.1
training:
  epochs: 1
  lr: 2e-5
  batch_size: 1
  lora:
    r: 8
    alpha: 16
  quantization: 4bit
output: {tmp_path / 'output'}
""",
        encoding="utf-8",
    )
    return config_path
