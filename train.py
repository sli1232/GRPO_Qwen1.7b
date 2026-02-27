"""
GRPO training script to improve Qwen 1.7B math ability.

Model  : Qwen/Qwen2.5-Math-1.5B
Dataset: BytedTsinghua-SIA/DAPO-Math-17k
Trainer: TRL GRPOTrainer
"""

import re

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

try:
    from math_verify import parse, verify
except ImportError:
    parse = None
    verify = None

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_NAME = "Qwen/Qwen2.5-Math-1.5B"
DATASET_NAME = "BytedTsinghua-SIA/DAPO-Math-17k"
OUTPUT_DIR = "output/qwen-math-grpo"

SYSTEM_PROMPT = (
    "Please provide your final answer within \\boxed{}."
)

# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------


def extract_ground_truth(row: dict) -> str:
    value = row.get("ground_truth", "")
    return value.strip() if isinstance(value, str) else ""




def make_prompt(example: dict) -> dict:
    """Convert a dataset row into the chat-message format expected by GRPOTrainer."""
    if isinstance(example.get("prompt"), list):
        problem = " ".join([msg.get("content", "") for msg in example.get("prompt", [])])
    else:
        problem = example.get("prompt") or example.get("problem") or ""

    answer = extract_ground_truth(example)
    if not answer and isinstance(example.get("reward_model"), dict):
        answer = str(example.get("reward_model", {}).get("ground_truth", "")).strip()
    if not answer and isinstance(example.get("reward_model"), str):
        answer = example.get("reward_model", "").strip()
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": problem},
        ],
        "answer": answer,
    }


# ---------------------------------------------------------------------------
# Reward functions
# ---------------------------------------------------------------------------

def _completion_to_text(completion) -> str:
    # completion is a list of chat messages: [{"role": "...", "content": "..."}]
    if isinstance(completion, list):
        for msg in completion:
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                return str(msg.get("content", ""))
        # fallback: concatenate any content fields
        return " ".join(str(m.get("content", "")) for m in completion if isinstance(m, dict))
    return str(completion)


def answers_match(prediction: str, ground_truth: str) -> bool:
    if parse is not None and verify is not None:
        try:
            pred_parsed = parse(prediction)
            gt_parsed = parse(ground_truth)
            return bool(verify(pred_parsed, gt_parsed))
        except Exception:
            pass

    return prediction.strip() == ground_truth.strip()

def accuracy_reward(completions: list[str], answer: list[str], **kwargs) -> list[float]:
    """Reward +1 if the model answer matches ground-truth via math_verify, else 0."""
    texts = [_completion_to_text(c) for c in completions]
    rewards = []
    for completion, gt in zip(texts, answer):
        rewards.append(1.0 if answers_match(completion, gt) else 0.0)
    return rewards

def format_reward(completions: list[str], **kwargs) -> list[float]:
    """Reward +0.5 if the completion contains \\boxed{...}."""
    texts = [_completion_to_text(c) for c in completions]
    rewards = []
    boxed_pattern = re.compile(r"\\boxed\{.+?\}", re.DOTALL)
    for completion in texts:
        has_boxed = bool(boxed_pattern.search(completion))
        rewards.append(0.5 if has_boxed else 0.0)
    return rewards


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    # ---- Model & tokenizer -------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    # Ensure tokenizer has a pad token for batching; use eos_token if not defined.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        trust_remote_code=True,
    )

    # ---- Dataset ------------------------------------------------------------
    dataset = load_dataset(DATASET_NAME, split="train")
    dataset = dataset.map(make_prompt, remove_columns=dataset.column_names)
    # use a small subset for demonstration; remove .select(...) for full training
    dataset = dataset.select(range(500))

    # ---- GRPO training config -----------------------------------------------
    training_args = GRPOConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=3,
        learning_rate=1e-6,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        # GRPO-specific
        max_completion_length=512,
        num_generations=6,
        temperature=1.0,
        # Disable vLLM for broad compatibility; set use_vllm=True to speed up
        # generation if vLLM is installed.
        use_vllm=False,
    )

    # ---- Trainer ------------------------------------------------------------
    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        reward_funcs=[accuracy_reward, format_reward],
    )

    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Training complete. Model saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
