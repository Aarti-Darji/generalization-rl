"""GRPO from the SFT model with TRL's GRPOTrainer.

Seeds change only the random state (data order, sampling); data and
hyperparameters stay fixed across runs.
"""

from __future__ import annotations

from pathlib import Path

from genrl.config import model_path
from genrl.data.datasets import format_prompt, load_problems
from genrl.training.rewards import exact_match_reward


def build_grpo_dataset(name: str):
    from datasets import Dataset

    return Dataset.from_list(
        [{"prompt": format_prompt(p), "answer": p.answer} for p in load_problems(name)]
    )


def train_grpo(cfg: dict, seed: int) -> Path:
    from trl import GRPOConfig, GRPOTrainer

    out = Path(cfg["output_dir"]) / f"seed{seed}"
    args = GRPOConfig(
        output_dir=str(out),
        seed=seed,
        data_seed=seed,
        max_steps=cfg["max_steps"],
        save_steps=cfg["save_steps"],
        save_only_model=True,
        learning_rate=cfg["learning_rate"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        num_generations=cfg["num_generations"],
        temperature=cfg["temperature"],
        max_completion_length=cfg["max_completion_length"],
        beta=cfg["beta"],
        use_vllm=cfg["use_vllm"],
        vllm_mode=cfg["vllm_mode"],
        vllm_gpu_memory_utilization=cfg["vllm_gpu_memory_utilization"],
        logging_steps=cfg["logging_steps"],
        report_to=cfg["report_to"],
        run_name=f"grpo-seed{seed}",
        bf16=True,
    )
    trainer = GRPOTrainer(
        model=model_path(cfg),
        reward_funcs=exact_match_reward,
        args=args,
        train_dataset=build_grpo_dataset(cfg["dataset"]),
    )
    trainer.train()
    trainer.save_model(str(out / "final"))
    return out
