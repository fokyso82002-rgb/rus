"""Quality of a trained adapter: mean loss on the answer tokens of an eval file.

    python harness/eval_adapter.py --adapter runs/solution_1/adapter --eval-file data/eval_public.jsonl

This is the same function the jury runs on its hidden rows. It loads the adapter with plain
transformers + peft (your solution is NOT imported), so the adapter must be a standard PEFT LoRA
adapter. Lower is better.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import MODEL  # noqa: E402

MAX_LENGTH = 512


def evaluate(adapter, eval_file):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tok = AutoTokenizer.from_pretrained(MODEL)
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_use_double_quant=True,
                               bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=quant,
                                                 device_map={"": 0}, dtype=torch.float16)
    if adapter != "none":
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    model.eval()

    total_nll, total_tokens, skipped = 0.0, 0, 0
    with open(eval_file, encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    for row in rows:
        # The same user turn Soup builds from an alpaca row (soup_cli/data/formats.py).
        extra = row.get("input") or ""
        user = f"{row['instruction']}\n{extra}".strip() if extra else row["instruction"]
        messages = [{"role": "user", "content": user}]
        prompt_ids = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                             return_dict=False)
        full_ids = tok.apply_chat_template(
            messages + [{"role": "assistant", "content": row["output"]}], tokenize=True,
            return_dict=False)[:MAX_LENGTH]
        if len(prompt_ids) >= len(full_ids):  # no answer token fits in the window
            skipped += 1
            continue
        n_prompt = len(prompt_ids)
        ids = torch.tensor([full_ids], device="cuda")
        labels = ids.clone()
        labels[:, :n_prompt] = -100
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            logits = model(input_ids=ids).logits.float()
        shift_logits, shift_labels = logits[:, :-1], labels[:, 1:]
        loss = torch.nn.functional.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1),
            ignore_index=-100, reduction="sum")
        total_nll += float(loss)
        total_tokens += int((shift_labels != -100).sum())
    if not total_tokens:
        raise SystemExit(f"no answer tokens to score in {eval_file}")
    return {"adapter": str(adapter), "eval_file": str(eval_file), "rows": len(rows),
            "skipped_rows": skipped, "tokens": total_tokens,
            "mean_loss": round(total_nll / total_tokens, 5)}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--adapter", required=True, help="adapter directory, or 'none'")
    parser.add_argument("--eval-file", required=True)
    parser.add_argument("--out", help="write the result here as JSON")
    args = parser.parse_args()
    result = evaluate(args.adapter, args.eval_file)
    print("EVAL " + json.dumps(result, ensure_ascii=False))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
