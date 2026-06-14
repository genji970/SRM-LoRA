from __future__ import annotations

from pathlib import Path
from typing import Any


class BaseMethod:
    name = "base"
    needs_hallucinations = False

    def __init__(self, env_path: str = "env.sh") -> None:
        self.env = self.load_env(env_path)

    @staticmethod
    def load_env(path: str) -> dict[str, str]:
        env: dict[str, str] = {}
        for raw in Path(path).read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export ") :]
            key, value = line.split("=", 1)
            env[key.strip()] = value.strip().strip("\"'")
        return env

    def apply(self, model: Any) -> Any:
        return model

    def check_dependencies(self) -> None:
        return None

    def loss(self, tokenizer: Any, max_length: int | None) -> Any:
        from train.loss import GoldCELoss

        return GoldCELoss(tokenizer=tokenizer, max_length=max_length)

    def before_optimizer_step(self, model: Any, step: int) -> dict[str, Any] | None:
        return None


class LoraMethod(BaseMethod):
    def __init__(self, env_path: str = "env.sh") -> None:
        super().__init__(env_path)
        self.r = int(self.env["LORA_R"])
        self.alpha = int(self.env["LORA_ALPHA"])
        self.dropout = float(self.env["LORA_DROPOUT"])
        self.bias = self.env["LORA_BIAS"]
        self.task_type = self.env["LORA_TASK_TYPE"]
        self.target_modules = [name.strip() for name in self.env["LORA_TARGET_MODULES"].split(",") if name.strip()]

    def apply(self, model: Any) -> Any:
        self.check_dependencies()
        from peft import LoraConfig, get_peft_model

        config = LoraConfig(
            r=self.r,
            lora_alpha=self.alpha,
            lora_dropout=self.dropout,
            bias=self.bias,
            task_type=self.task_type,
            target_modules=self.target_modules,
        )
        return get_peft_model(model, config)

    def check_dependencies(self) -> None:
        try:
            import peft  # noqa: F401
        except ImportError as exc:
            raise ImportError("peft is required for LoRA methods. Install it with: pip install peft") from exc
