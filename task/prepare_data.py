"""Download the task data and write the public files.

    python task/prepare_data.py            # -> data/train.jsonl, data/eval_public.jsonl

Data: AmanMussa/kazakh-instruction-v2 (Hugging Face, MIT license), Kazakh instructions in the
alpaca format (instruction, input, output). The jury trains and evaluates on other, hidden rows
of the same dataset, so do not tune anything to these particular rows.
"""

import argparse
import json
from pathlib import Path

DATASET = "AmanMussa/kazakh-instruction-v2"
REVISION = "c641407ee50723c84a43ea2ec867b464d3893c12"
TRAIN_ROWS = range(0, 256)
EVAL_ROWS = range(40000, 40256)


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps({"instruction": row["instruction"], "input": row["input"],
                                     "output": row["output"]}, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "data"))
    args = parser.parse_args()

    from datasets import load_dataset

    dataset = load_dataset(DATASET, split="train", revision=REVISION)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "train.jsonl", (dataset[i] for i in TRAIN_ROWS))
    write_jsonl(out / "eval_public.jsonl", (dataset[i] for i in EVAL_ROWS))
    print(f"wrote {len(TRAIN_ROWS)} training rows and {len(EVAL_ROWS)} eval rows to {out}")


if __name__ == "__main__":
    main()
