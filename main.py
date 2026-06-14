from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from model.pipeline import Pipeline
from train.train import Trainer


# Extra eval aliases used by ablation scripts. These are kept here so the
# ablation runner can use EVAL_DATASET=drop,hotpotqa_fullwiki,... even when the
# existing Pipeline.EVAL_SOURCES table has not been updated yet.
EVAL_SOURCE_OVERRIDES: dict[str, tuple[str, str, str]] = {
    "drop": ("ucinlp/drop", "", "validation"),
    "hotpotqa_fullwiki": ("hotpotqa/hotpot_qa", "fullwiki", "validation"),
    "halueval_dialogue": ("pminervini/HaluEval", "dialogue", "data"),
    "halueval_summarization": ("pminervini/HaluEval", "summarization", "data"),
}


def load_env(path: str = "env.sh") -> dict[str, str]:
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


def optional_int(value: str) -> int | None:
    return None if value.lower() in {"", "none", "null"} else int(value)


def bool_value(value: str) -> bool:
    lowered = value.lower()
    if lowered in {"true", "1", "yes"}:
        return True
    if lowered in {"false", "0", "no"}:
        return False
    raise ValueError(f"expected bool value, got {value}")


def csv_values(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def set_global_seed(seed: int) -> None:
    import random

    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def dataset_records(dataset: str, subset: str, split: str, max_samples: int | None) -> list[Mapping[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("datasets is required to load train/eval records") from exc

    print(f"[main] loading dataset={dataset} subset={subset or '-'} split={split}", flush=True)
    loaded = load_dataset(dataset, subset, split=split) if subset else load_dataset(dataset, split=split)
    if max_samples is not None:
        loaded = loaded.select(range(min(max_samples, len(loaded))))
    records = [dict(row) for row in loaded]
    print(f"[main] loaded {len(records)} records", flush=True)
    return records


def eval_source(name: str) -> tuple[str, str, str]:
    if name in EVAL_SOURCE_OVERRIDES:
        return EVAL_SOURCE_OVERRIDES[name]
    if name not in Pipeline.EVAL_SOURCES:
        names = sorted(set(Pipeline.EVAL_SOURCES) | set(EVAL_SOURCE_OVERRIDES))
        raise ValueError(f"EVAL_DATASET must be one of: {', '.join(names)}")
    return Pipeline.EVAL_SOURCES[name]


def main() -> None:
    env = load_env()
    print("[main] start", flush=True)
    seed = int(env.get("SEED", env.get("TRAIN_SEED", env.get("CONTRASTIVE_SEED", "42"))))
    set_global_seed(seed)
    print(f"[main] seed={seed}", flush=True)

    experiments = csv_values(env.get("EXPERIMENTS", env.get("METHOD", "contrastive_lora")))
    print(f"[main] experiments={experiments}", flush=True)
    train_records = dataset_records(
        env["TRAIN_DATASET"],
        env["TRAIN_SUBSET"],
        env["TRAIN_SPLIT"],
        optional_int(env["MAX_TRAIN_SAMPLES"]),
    )
    eval_name = env["EVAL_DATASET"]
    eval_dataset, eval_subset, eval_split = eval_source(eval_name)
    print(f"[main] eval_name={eval_name}", flush=True)
    eval_records = dataset_records(
        eval_dataset,
        eval_subset,
        eval_split,
        optional_int(env["MAX_EVAL_SAMPLES"]),
    )

    for i, method_name in enumerate(experiments, 1):
        print(f"[main] experiment {i}/{len(experiments)} method={method_name} eval={eval_name}", flush=True)
        trainer = Trainer(train_records, eval_records, method_name=method_name)
        print("[main] trainer ready", flush=True)
        trainer.train(add_hallucinations=bool_value(env["ADD_HALLUCINATIONS"]))
        del trainer
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


if __name__ == "__main__":
    main()
