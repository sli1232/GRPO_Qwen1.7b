"""
Evaluate a trained Qwen math model against the base model on JSONL test sets.
"""

import argparse
import json
import os
import re
from typing import Iterable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SYSTEM_PROMPT = (
    "You are a helpful math assistant. "
    "Think step by step inside <think>...</think> tags, "
    "then provide the final answer inside \\boxed{...}."
)


def extract_boxed_content(text: str) -> str:
    """Extract the content of the last \\boxed{...} in *text*, handling nested braces."""
    results = []
    idx = 0
    while True:
        start = text.find(r"\\boxed{", idx)
        if start == -1:
            break
        depth = 0
        content_start = start + len(r"\\boxed{")
        for i in range(content_start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                if depth == 0:
                    results.append(text[content_start:i].strip())
                    idx = i + 1
                    break
                depth -= 1
        else:
            break
    return results[-1] if results else ""


def normalize_answer(text: str) -> str:
    """Normalize answers for robust exact-match comparison."""
    text = text.strip()
    if text.startswith("$") and text.endswith("$"):
        text = text[1:-1].strip()
    # Remove surrounding \boxed{...} if present.
    boxed = extract_boxed_content(text)
    if boxed:
        text = boxed
    # Remove whitespace to tolerate LaTeX spacing differences.
    text = re.sub(r"\s+", "", text)
    return text


def extract_ground_truth(row: dict) -> str:
    """Extract a comparable ground-truth answer from various JSONL schemas."""
    raw = row.get("ground_truth", row.get("answer", ""))
    if isinstance(raw, dict):
        # Common keys across math datasets.
        for key in ["ground_truth", "final", "answer", "value", "label"]:
            value = raw.get(key)
            if isinstance(value, str) and value.strip():
                return value
        return ""
    if isinstance(raw, list):
        for value in raw:
            if isinstance(value, str) and value.strip():
                return value
        return ""
    return str(raw) if raw is not None else ""


def load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def batched(iterable: Iterable, batch_size: int):
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def build_inputs(tokenizer, questions: list[str]):
    prompts = []
    for question in questions:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        prompts.append(prompt)
    return tokenizer(prompts, return_tensors="pt", padding=True)


def generate_answers(
    model,
    tokenizer,
    questions: list[str],
    max_new_tokens: int,
    temperature: float,
) -> list[str]:
    inputs = build_inputs(tokenizer, questions)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    do_sample = temperature > 0.0
    generated = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature if do_sample else None,
        top_p=1.0,
    )
    gen_tokens = generated[:, inputs["input_ids"].shape[1]:]
    return tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)


def evaluate_model(
    model_name: str,
    data_path: str,
    batch_size: int,
    max_new_tokens: int,
    temperature: float,
    max_samples: int | None,
) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        trust_remote_code=True,
        device_map="auto",
    )
    rows = load_jsonl(data_path)
    if max_samples is not None:
        rows = rows[:max_samples]

    correct = 0
    total = 0

    for batch in batched(rows, batch_size):
        questions = [row.get("question", "") for row in batch]
        gts = [extract_ground_truth(row) for row in batch]
        outputs = generate_answers(
            model,
            tokenizer,
            questions,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        for pred, gt in zip(outputs, gts):
            pred_norm = normalize_answer(pred)
            gt_norm = normalize_answer(gt)
            if pred_norm == gt_norm:
                correct += 1
            total += 1

    accuracy = correct / total if total else 0.0
    return {
        "model": model_name,
        "data": os.path.basename(data_path),
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Qwen math models.")
    parser.add_argument(
        "--model",
        default="output/qwen-math-grpo",
        help="Path/name of the trained model.",
    )
    parser.add_argument(
        "--base-model",
        default="Qwen/Qwen2.5-Math-1.5B",
        help="Path/name of the base model.",
    )
    parser.add_argument(
        "--data-dir",
        default="data/test",
        help="Directory containing JSONL test files.",
    )
    parser.add_argument(
        "--files",
        nargs="*",
        default=[
            "gsm8k.jsonl",
            "math.jsonl",
            "amc.jsonl",
            "aime.jsonl",
            "minerva.jsonl",
            "olympiad_bench.jsonl",
        ],
        help="List of JSONL files to evaluate.",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-samples", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    files = [os.path.join(args.data_dir, f) for f in args.files]

    for model_name in [args.base_model, args.model]:
        print(f"\nEvaluating: {model_name}")
        for path in files:
            if not os.path.exists(path):
                print(f"  Skipping missing file: {path}")
                continue
            metrics = evaluate_model(
                model_name,
                path,
                batch_size=args.batch_size,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                max_samples=args.max_samples,
            )
            acc = metrics["accuracy"] * 100
            print(
                f"  {metrics['data']}: {metrics['correct']}/{metrics['total']} "
                f"({acc:.2f}%)"
            )


if __name__ == "__main__":
    main()
