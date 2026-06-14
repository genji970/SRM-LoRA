from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any


def patch_torch_custom_op_annotations() -> None:
    import typing

    import torch

    if getattr(torch.library, "_string_annotation_patch", False):
        return

    original_custom_op = torch.library.custom_op

    def resolve_annotations(fn: Any) -> Any:
        try:
            fn.__annotations__ = typing.get_type_hints(fn, globalns={**fn.__globals__, "torch": torch})
        except Exception:
            fn.__annotations__ = {
                key: torch.Tensor if value == "torch.Tensor" else value
                for key, value in getattr(fn, "__annotations__", {}).items()
            }
        return fn

    def patched_custom_op(name: str, fn: Any = None, /, *args: Any, **kwargs: Any) -> Any:
        if fn is not None:
            return original_custom_op(name, resolve_annotations(fn), *args, **kwargs)
        decorator = original_custom_op(name, fn, *args, **kwargs)

        def wrapped(inner_fn: Any) -> Any:
            return decorator(resolve_annotations(inner_fn))

        return wrapped

    torch.library.custom_op = patched_custom_op
    torch.library._string_annotation_patch = True


class ModelLoad:
    def __init__(self, env_path: str = "env.sh") -> None:
        self.env = self.load_env(env_path)
        self.model_name = self.env["MODEL_NAME"]
        self.dtype = self.env.get("MODEL_DTYPE", "bfloat16")
        self.device_map = self.env.get("MODEL_DEVICE_MAP", "auto")
        self.low_cpu_mem_usage = self.bool_value(self.env.get("MODEL_LOW_CPU_MEM_USAGE", "true"))
        self.trust_remote_code = self.bool_value(self.env.get("MODEL_TRUST_REMOTE_CODE", "false"))

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

    def load_tokenizer(self) -> Any:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(self.model_name, **self.tokenizer_kwargs())
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
        return tokenizer

    def load_model(self) -> Any:
        patch_torch_custom_op_annotations()

        from transformers import AutoModelForCausalLM

        kwargs = self.model_kwargs()
        print(f"[model_load] loading model={self.model_name} kwargs={self.printable_kwargs(kwargs)}", flush=True)
        model = AutoModelForCausalLM.from_pretrained(self.model_name, **kwargs)
        model.eval()
        print(f"[model_load] loaded device={self.model_device(model)} dtype={self.model_dtype(model)}", flush=True)
        return model

    def model_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"low_cpu_mem_usage": self.low_cpu_mem_usage, "trust_remote_code": self.trust_remote_code}
        dtype = self.torch_dtype()
        if dtype is not None:
            kwargs["torch_dtype"] = dtype
        if self.device_map.lower() not in {"", "none", "null"}:
            kwargs["device_map"] = self.device_map
        return kwargs

    def tokenizer_kwargs(self) -> dict[str, Any]:
        return {"trust_remote_code": self.trust_remote_code}

    def torch_dtype(self) -> Any:
        if self.dtype.lower() in {"", "none", "null"}:
            return None
        if self.dtype.lower() == "auto":
            return "auto"

        import torch

        dtypes = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        if self.dtype.lower() not in dtypes:
            raise ValueError(f"MODEL_DTYPE must be one of: auto, bfloat16, float16, float32")
        return dtypes[self.dtype.lower()]

    @staticmethod
    def bool_value(value: str) -> bool:
        lowered = value.lower()
        if lowered in {"true", "1", "yes"}:
            return True
        if lowered in {"false", "0", "no"}:
            return False
        raise ValueError(f"expected bool value, got {value}")

    @staticmethod
    def model_device(model: Any) -> str:
        try:
            return str(next(model.parameters()).device)
        except StopIteration:
            return "unknown"

    @staticmethod
    def model_dtype(model: Any) -> str:
        try:
            return str(next(model.parameters()).dtype)
        except StopIteration:
            return "unknown"

    @staticmethod
    def printable_kwargs(kwargs: Mapping[str, Any]) -> dict[str, str]:
        return {key: str(value) for key, value in kwargs.items()}

    def load(self) -> tuple[Any, Any]:
        tokenizer = self.load_tokenizer()
        model = self.load_model()
        return model, tokenizer


if __name__ == "__main__":
    loader = ModelLoad()
    print({"model_name": loader.model_name, "tokenizer_padding_side": "left", "trust_remote_code": loader.trust_remote_code})
