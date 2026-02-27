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

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_NAME = "Qwen/Qwen2.5-Math-1.5B"
DATASET_NAME = "BytedTsinghua-SIA/DAPO-Math-17k"
OUTPUT_DIR = "output/qwen-math-grpo"

SYSTEM_PROMPT = (
    "You are a helpful math assistant. "
    "Think step by step inside <think>...</think> tags, "
    "then provide the final answer inside \\boxed{...}."
)

# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------


def extract_boxed_content(text: str) -> str:
    """Extract the content of the last \\boxed{...} in *text*, handling nested braces."""
    results = []
    idx = 0
    while True:
        start = text.find(r"\boxed{", idx)
        if start == -1:
            break
        # Walk forward to find the matching closing brace.
        depth = 0
        content_start = start + len(r"\boxed{")
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
            break  # unmatched brace – stop
    return results[-1] if results else ""




def make_prompt(example: dict) -> dict:
    """Convert a dataset row into the chat-message format expected by GRPOTrainer."""
    if isinstance(example.get("prompt"), list):
        problem = " ".join([msg.get("content", "") for msg in example.get("prompt", [])])
    else:
        problem = example.get("prompt") or example.get("problem") or ""

    if isinstance(example.get("reward_model"), dict):
        answer = example.get("reward_model", {}).get("ground_truth", "")
    else:
        answer = example.get("reward_model") or extract_boxed_content(example.get("solution", ""))
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

def accuracy_reward(completions: list[str], answer: list[str], **kwargs) -> list[float]:
    """Reward +1 if the model's boxed answer matches the ground-truth, else 0."""
    texts = [_completion_to_text(c) for c in completions]
    rewards = []
    for completion, gt in zip(texts, answer):
        predicted = extract_boxed_content(completion)
        rewards.append(1.0 if predicted == gt else 0.0)
    return rewards

def format_reward(completions: list[str], **kwargs) -> list[float]:
    """Reward +0.5 if the completion contains <think>…</think> with content and \\boxed{...}."""
    texts = [_completion_to_text(c) for c in completions]
    rewards = []
    think_pattern = re.compile(r"<think>.+?</think>", re.DOTALL)
    for completion in texts:
        has_think = bool(think_pattern.search(completion))
        has_boxed = bool(extract_boxed_content(completion))
        rewards.append(0.5 if (has_think and has_boxed) else 0.0)
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
