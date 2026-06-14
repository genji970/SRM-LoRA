from __future__ import annotations

from model.method.base import BaseMethod
from model.method.contrastive_lora.main import ContrastiveLora
from model.method.plain_lora.main import PlainLora
from model.method.sr_lora.main import SubRiemannianLora

METHODS = {
    "none": BaseMethod,
    "plain_lora": PlainLora,
    "contrastive_lora": ContrastiveLora,
    "sr_lora": SubRiemannianLora,
}


def load_method(env_path: str = "env.sh") -> BaseMethod:
    env = BaseMethod.load_env(env_path)
    name = env.get("METHOD", "contrastive_lora")
    if name not in METHODS:
        valid = ", ".join(sorted(METHODS))
        raise ValueError(f"METHOD must be one of: {valid}")
    return METHODS[name](env_path)


__all__ = ["BaseMethod", "PlainLora", "ContrastiveLora", "SubRiemannianLora", "load_method"]
