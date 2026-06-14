from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any


class GoldHallLoss:
    @dataclass(frozen=True)
    class Output:
        loss: Any
        gold_ce: Any
        hall_ce: Any | None
        loss_fn: Any | None = None
        samples: Sequence[Any] | None = None

    def __init__(
        self,
        tokenizer: Any,
        max_length: int | None,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, model: Any, samples: Sequence[Any]) -> Output:
        gold_ce = self.sequence_ce(model, samples, "answer")
        hall_samples = [sample for sample in samples if getattr(sample, "hallucinated_answer", "")]
        hall_ce = self.sequence_ce(model, hall_samples, "hallucinated_answer") if hall_samples else None
        loss = gold_ce
        if hall_ce is not None:
            loss = loss - hall_ce
        return self.Output(loss=loss, gold_ce=gold_ce, hall_ce=hall_ce, loss_fn=self, samples=tuple(samples))

    def sequence_ce(self, model: Any, samples: Sequence[Any], answer_attr: str) -> Any:
        import torch.nn.functional as F

        batch = self.build_lm_batch(samples, answer_attr)
        batch = {key: value.to(model.device) for key, value in batch.items()}
        logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = batch["labels"][:, 1:].contiguous()
        return F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )

    def build_lm_batch(self, samples: Sequence[Any], answer_attr: str) -> dict[str, Any]:
        import torch

        rows = [self.encode_sample(sample, getattr(sample, answer_attr)) for sample in samples]
        pad_id = self.tokenizer.pad_token_id
        input_ids = self.pad([row["input_ids"] for row in rows], pad_id)
        attention_mask = self.pad([row["attention_mask"] for row in rows], 0)
        labels = self.pad([row["labels"] for row in rows], -100)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    def encode_sample(self, sample: Any, answer: str) -> dict[str, list[int]]:
        prompt = self.prompt(sample)
        prompt_ids = self.tokenizer(prompt, add_special_tokens=True)["input_ids"]
        answer_ids = self.tokenizer(answer, add_special_tokens=False)["input_ids"]
        if self.tokenizer.eos_token_id is not None:
            answer_ids = answer_ids + [self.tokenizer.eos_token_id]
        full_ids = prompt_ids + answer_ids
        labels = [-100] * len(prompt_ids) + answer_ids
        if self.max_length is not None:
            full_ids = full_ids[-self.max_length :]
            labels = labels[-self.max_length :]
        return {
            "input_ids": full_ids,
            "attention_mask": [1] * len(full_ids),
            "labels": labels,
        }

    @staticmethod
    def prompt(sample: Any) -> str:
        return (
            "Answer the question using the reference.\n\n"
            f"Reference:\n{sample.reference}\n\n"
            f"Question:\n{sample.question}\n\n"
            "Answer:"
        )

    def pad(self, rows: Sequence[list[int]], pad_value: int) -> list[list[int]]:
        width = max(len(row) for row in rows)
        if self.tokenizer.padding_side == "left":
            return [[pad_value] * (width - len(row)) + row for row in rows]
        return [row + [pad_value] * (width - len(row)) for row in rows]


class GoldCELoss(GoldHallLoss):
    def __call__(self, model: Any, samples: Sequence[Any]) -> GoldHallLoss.Output:
        gold_ce = self.sequence_ce(model, samples, "answer")
        return self.Output(loss=gold_ce, gold_ce=gold_ce, hall_ce=None, loss_fn=self, samples=tuple(samples))
