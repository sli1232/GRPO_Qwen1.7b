# GRPO_Qwen1.7b

Fine-tune **Qwen2.5-Math-1.5B** on the [DAPO-Math-17k](https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k) dataset using the **TRL GRPOTrainer** to improve the model's mathematical reasoning ability.

---

## Overview

| Item | Value |
|---|---|
| Base model | `Qwen/Qwen2.5-Math-1.5B` |
| Dataset | `BytedTsinghua-SIA/DAPO-Math-17k` |
| Training algorithm | GRPO (Group Relative Policy Optimization) |
| Framework | [TRL](https://github.com/huggingface/trl) ≥ 0.13 |

The model is prompted to reason inside `<think>…</think>` tags before producing a `\boxed{…}` answer.  
Two reward functions are used:

- **accuracy_reward** (+1.0) — the predicted `\boxed{}` answer matches the ground-truth answer.  
- **format_reward** (+0.5) — the completion contains both `<think>…</think>` and `\boxed{}`.

---

## Setup

```bash
git clone https://github.com/sli1232/GRPO_Qwen1.7b.git
cd GRPO_Qwen1.7b
pip install -r requirements.txt
```

> **GPU requirements**: at least one GPU with ~24 GB VRAM is recommended
> (e.g. A100 40 GB). Reduce `per_device_train_batch_size` or enable
> `gradient_checkpointing` for smaller GPUs.

---

## Training

```bash
# Single GPU
python train.py

# Multi-GPU with Accelerate
accelerate launch --num_processes <N> train.py
```

Checkpoints and the final model are saved to `output/qwen-math-grpo/`.

### Key hyper-parameters (edit in `train.py`)

| Parameter | Default | Description |
|---|---|---|
| `num_train_epochs` | 1 | Number of full passes over the dataset |
| `per_device_train_batch_size` | 2 | Samples per GPU per step |
| `gradient_accumulation_steps` | 4 | Effective batch size multiplier |
| `learning_rate` | 1e-6 | Peak learning rate |
| `num_generations` | 8 | GRPO rollout samples per prompt |
| `max_prompt_length` | 512 | Max tokens for the input prompt |
| `max_completion_length` | 2048 | Max tokens the model may generate |
| `use_vllm` | False | Enable vLLM for faster generation |

---

## File structure

```
.
├── train.py          # Main GRPO training script
├── requirements.txt  # Python dependencies
└── README.md
```
