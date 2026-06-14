from __future__ import annotations

from typing import Any

from model.method.base import LoraMethod


class ContrastiveLora(LoraMethod):
    name = "contrastive_lora"
    needs_hallucinations = True

    def loss(self, tokenizer: Any, max_length: int | None) -> Any:
        from model.method.contrastive_lora.loss import ContrastiveLoraLoss

        return ContrastiveLoraLoss(tokenizer=tokenizer, max_length=max_length)
