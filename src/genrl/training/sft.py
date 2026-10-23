"""Short SFT on GSM8K train solutions with TRL's SFTTrainer."""

from __future__ import annotations

from genrl.config import model_path
from genrl.data.datasets import SYSTEM_PROMPT, load_problems


def _gsm8k_to_messages(solution: str) -> str:
    # Rewrite the trailing "#### x" as \boxed{x} so SFT matches the eval format.
    body, _, ans = solution.rpartition("####")
    return f"{body.strip()}\nThe answer is \\boxed{{{ans.strip().replace(',', '')}}}."


def build_sft_dataset(name: str):
    """Conversational prompt/completion rows, so the loss covers only the answer."""
    from datasets import Dataset

    problems = load_problems(name)
    return Dataset.from_list(
        [
            {
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": p.question},
                ],
                "completion": [
                    {"role": "assistant", "content": _gsm8k_to_messages(p.solution)},
                ],
            }
            for p in problems
        ]
    )


def train_sft(cfg: dict) -> None:
    from trl import SFTConfig, SFTTrainer

    args = SFTConfig(
        output_dir=cfg["output_dir"],
        seed=cfg["seed"],
        num_train_epochs=cfg["num_train_epochs"],
        learning_rate=cfg["learning_rate"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        warmup_ratio=cfg["warmup_ratio"],
        lr_scheduler_type=cfg["lr_scheduler_type"],
        logging_steps=cfg["logging_steps"],
        save_strategy=cfg["save_strategy"],
        report_to=cfg["report_to"],
        completion_only_loss=True,  # mask system + question tokens out of the loss
        bf16=True,
    )
    trainer = SFTTrainer(
        model=model_path(cfg), args=args, train_dataset=build_sft_dataset(cfg["dataset"])
    )
    trainer.train()
    trainer.save_model(cfg["output_dir"])
