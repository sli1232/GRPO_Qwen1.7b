"""
Evaluate a trained Qwen math model against the base model on JSONL test sets.
"""

import argparse
import json
import os
from typing import Iterable

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from math_verify import parse, verify
except ImportError:
    parse = None
    verify = None

SYSTEM_PROMPT = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)


def _extract_text_values(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, (int, float, bool)):
        return [str(value)]
    if isinstance(value, list):
        results = []
        for item in value:
            results.extend(_extract_text_values(item))
        return results
    if isinstance(value, dict):
        results = []
        preferred_keys = ["ground_truth", "final", "answer", "value", "label", "solution"]
        for key in preferred_keys:
            if key in value:
                results.extend(_extract_text_values(value.get(key)))
        if not results:
            for item in value.values():
                results.extend(_extract_text_values(item))
        return results
    return []


def extract_ground_truth_candidates(row: dict) -> list[str]:
    candidates = []
    for key in ["ground_truth", "answer", "solution"]:
        candidates.extend(_extract_text_values(row.get(key)))

    deduped = []
    seen = set()
    for candidate in candidates:
        if candidate not in seen:
            seen.add(candidate)
            deduped.append(candidate)

    return deduped or [""]


def answers_match(prediction: str, ground_truth_candidates: list[str]) -> tuple[bool, str]:
    if parse is not None and verify is not None:
        try:
            prediction_parsed = parse(prediction)
            for candidate in ground_truth_candidates:
                try:
                    candidate_parsed = parse(candidate)
                except Exception:
                    continue
                if bool(verify(prediction_parsed, candidate_parsed)):
                    return True, candidate
        except Exception:
            pass

    prediction_text = prediction.strip()
    for candidate in ground_truth_candidates:
        if prediction_text == candidate.strip():
            return True, candidate

    return False, (ground_truth_candidates[0] if ground_truth_candidates else "")


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
        pad_token_id=tokenizer.pad_token_id,
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
    output_handle,
) -> dict:
    model_label = os.path.basename(model_name.rstrip("/")) or model_name
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
    total_rows = len(rows)
    with tqdm(
        total=total_rows,
        desc=f"{model_label} | {os.path.basename(data_path)}",
        unit="ex",
        leave=False,
    ) as pbar:
        for batch in batched(rows, batch_size):
            questions = [row.get("question", "") for row in batch]
            gt_candidates_batch = [extract_ground_truth_candidates(row) for row in batch]
            outputs = generate_answers(
                model,
                tokenizer,
                questions,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
            batch_lines = []
            for row, question, pred, gt_candidates in zip(batch, questions, outputs, gt_candidates_batch):
                is_correct, matched_gt = answers_match(pred, gt_candidates)
                if is_correct:
                    correct += 1
                total += 1

                output_record = {
                    "record_type": "prediction",
                    "model": model_name,
                    "data": os.path.basename(data_path),
                    "question": question,
                    "ground_truth": matched_gt,
                    "ground_truth_candidates": gt_candidates,
                    "prediction": pred,
                    "is_correct": is_correct,
                }
                if "id" in row:
                    output_record["id"] = row["id"]
                batch_lines.append(json.dumps(output_record, ensure_ascii=False) + "\n")

            output_handle.writelines(batch_lines)
            pbar.update(len(batch))

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
    parser.add_argument(
        "--output-jsonl",
        default="output/eval_results.jsonl",
        help="Path to save prediction records and summary metrics in JSONL format.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    files = [os.path.join(args.data_dir, f) for f in args.files]
    output_dir = os.path.dirname(args.output_jsonl)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(args.output_jsonl, "w", encoding="utf-8") as output_handle:
        run_config = {
            "record_type": "run_config",
            "model": args.model,
            "base_model": args.base_model,
            "data_dir": args.data_dir,
            "files": args.files,
            "batch_size": args.batch_size,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "max_samples": args.max_samples,
        }
        output_handle.write(json.dumps(run_config, ensure_ascii=False) + "\n")

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
                    output_handle=output_handle,
                )
                output_handle.write(
                    json.dumps(
                        {
                            "record_type": "summary",
                            "model": metrics["model"],
                            "data": metrics["data"],
                            "total": metrics["total"],
                            "correct": metrics["correct"],
                            "accuracy": metrics["accuracy"],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                acc = metrics["accuracy"] * 100
                print(
                    f"  {metrics['data']}: {metrics['correct']}/{metrics['total']} "
                    f"({acc:.2f}%)"
                )

    print(f"\nSaved JSONL output to: {args.output_jsonl}")


if __name__ == "__main__":
    main()
