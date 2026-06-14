from __future__ import annotations

from typing import Any

from model.method.base import LoraMethod


class PlainLora(LoraMethod):
    name = "plain_lora"

    def loss(self, tokenizer: Any, max_length: int | None) -> Any:
        from model.method.plain_lora.loss import PlainLoraLoss

        return PlainLoraLoss(tokenizer=tokenizer, max_length=max_length)
