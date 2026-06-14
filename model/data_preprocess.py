from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from importlib import import_module
from pathlib import Path
from random import Random
import sys
from typing import Any

sys.path.append(str(Path(__file__).resolve().parents[1]))


class DataPreprocess:
    @dataclass(frozen=True)
    class Sample:
        question: str
        reference: str
        answer: str
        hallucinated_answer: str = ""

        def as_dict(self) -> dict[str, str]:
            data = {"question": self.question, "reference": self.reference, "answer": self.answer}
            if self.hallucinated_answer:
                data["hallucinated_answer"] = self.hallucinated_answer
            return data

    @dataclass(frozen=True)
    class Batch:
        samples: tuple[Sample, ...]

        def raw(self) -> list[dict[str, str]]:
            return [sample.as_dict() for sample in self.samples]

    def __init__(
        self,
        dataset: str,
        subset: str,
        split: str,
        records: Sequence[Mapping[str, Any]],
        env_path: str = "env.sh",
        sample_limit: int | None = None,
        batch_size: int | None = None,
    ) -> None:
        env = self.load_env(env_path)
        self.model_name = env["MODEL_NAME"]
        self.batch_size = batch_size or int(env["CONTRASTIVE_BATCH_SIZE"])
        self.max_length = self.optional_int(env["CONTRASTIVE_MAX_LENGTH"])
        self.shuffle = self.bool_value(env["CONTRASTIVE_SHUFFLE"])
        self.drop_last = self.bool_value(env["CONTRASTIVE_DROP_LAST"])
        self.seed = int(env["CONTRASTIVE_SEED"])
        self.padding = self.bool_or_str(env["CONTRASTIVE_PADDING"])
        self.truncation = self.bool_value(env["CONTRASTIVE_TRUNCATION"])
        self.return_tensors = self.optional_str(env["CONTRASTIVE_RETURN_TENSORS"])
        self.samples = self.limit_samples(self.load_samples(dataset, subset, split, records), sample_limit)

    @staticmethod
    def limit_samples(samples: list[Sample], sample_limit: int | None) -> list[Sample]:
        return samples if sample_limit is None else samples[:sample_limit]

    @classmethod
    def load_samples(cls, dataset: str, subset: str, split: str, records: Sequence[Mapping[str, Any]]) -> list[Sample]:
        module = import_module("data_load.data_load")
        data_cls = getattr(module, "DataLoad", None) or getattr(module, "ContrastiveData", None)
        if data_cls is None:
            raise ImportError("data_load.data_load must define DataLoad or ContrastiveData")

        loaded = data_cls(dataset, subset, split, records)
        raw_samples = loaded.samples if hasattr(loaded, "samples") else loaded.as_list()
        return [cls.to_sample(sample) for sample in raw_samples]

    @classmethod
    def to_sample(cls, sample: Any) -> Sample:
        if isinstance(sample, Mapping):
            return cls.Sample(
                cls.text(sample["question"]),
                cls.text(sample["reference"]),
                cls.text(sample["answer"]),
                cls.text(sample.get("hallucinated_answer", "")),
            )
        return cls.Sample(
            cls.text(sample.question),
            cls.text(sample.reference),
            cls.text(sample.answer),
            cls.text(getattr(sample, "hallucinated_answer", "")),
        )

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

    @staticmethod
    def bool_value(value: str) -> bool:
        lowered = value.lower()
        if lowered in {"true", "1", "yes"}:
            return True
        if lowered in {"false", "0", "no"}:
            return False
        raise ValueError(f"expected bool value, got {value}")

    @classmethod
    def bool_or_str(cls, value: str) -> bool | str:
        lowered = value.lower()
        if lowered in {"true", "1", "yes", "false", "0", "no"}:
            return cls.bool_value(value)
        return value

    @staticmethod
    def optional_int(value: str) -> int | None:
        return None if value.lower() in {"", "none", "null"} else int(value)

    @staticmethod
    def optional_str(value: str) -> str | None:
        return None if value.lower() in {"", "none", "null"} else value

    @staticmethod
    def text(value: str) -> str:
        return " ".join(value.strip().split())

    def tokenizer_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "padding": self.padding,
            "truncation": self.truncation,
            "max_length": self.max_length,
        }
        if self.return_tensors is not None:
            kwargs["return_tensors"] = self.return_tensors
        return kwargs

    def load_tokenizer(self) -> Any:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(self.model_name)

    def batches(self) -> Iterator[Batch]:
        indices = list(range(len(self.samples)))
        if self.shuffle:
            Random(self.seed).shuffle(indices)

        batch = []
        for i in indices:
            batch.append(self.samples[i])
            if len(batch) == self.batch_size:
                yield self.Batch(tuple(batch))
                batch.clear()

        if batch and not self.drop_last:
            yield self.Batch(tuple(batch))

    def add_hallucinated_answer(self, sample_index: int, prediction: str) -> None:
        self.samples[sample_index] = replace(
            self.samples[sample_index],
            hallucinated_answer=self.text(prediction),
        )

    def model_inputs(self, batch: Batch, tokenizer: Any | None = None) -> dict[str, Any]:
        tokenizer = tokenizer or self.load_tokenizer()
        kwargs = self.tokenizer_kwargs()
        inputs = {
            "question": tokenizer([sample.question for sample in batch.samples], **kwargs),
            "reference": tokenizer([sample.reference for sample in batch.samples], **kwargs),
            "answer": tokenizer([sample.answer for sample in batch.samples], **kwargs),
        }
        if any(sample.hallucinated_answer for sample in batch.samples):
            inputs["hallucinated_answer"] = tokenizer(
                [sample.hallucinated_answer for sample in batch.samples],
                **kwargs,
            )
        return inputs


if __name__ == "__main__":
    records = [
        {
            "knowledge": "LoRA freezes the base model and trains small adapter weights.",
            "question": "What does LoRA train?",
            "right_answer": "LoRA trains small adapter weights.",
        },
        {
            "knowledge": "Contrastive learning pulls similar pairs closer.",
            "question": "What does contrastive learning do to similar pairs?",
            "right_answer": "It pulls similar pairs closer.",
        },
    ]

    data = DataPreprocess("pminervini/HaluEval", "qa", "data", records)
    for batch in data.batches():
        print(batch.raw())
