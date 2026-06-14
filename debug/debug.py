from __future__ import annotations

from collections.abc import Mapping, Sequence
import argparse
import csv
import json
from pathlib import Path
import sys
from time import strftime
from typing import Any

sys.path.append(str(Path(__file__).resolve().parents[3]))

from model.data_preprocess import DataPreprocess
from model.model_load import ModelLoad, patch_torch_custom_op_annotations
from model.pipeline import EvalAnswerJudge, Pipeline


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


def optional_int(value: str | None) -> int | None:
    if value is None:
        return None
    return None if value.lower() in {"", "none", "null"} else int(value)


def bool_value(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    lowered = value.strip().lower()
    if lowered in {"true", "1", "yes", "y", "on"}:
        return True
    if lowered in {"false", "0", "no", "n", "off"}:
        return False
    return default


class SrMaskDebugRecorder:
    """File-only SR soft-mask diagnostics used during training."""

    def __init__(self, env: Mapping[str, str], method_name: str = "sr_lora") -> None:
        self.env = dict(env)
        self.method_name = method_name
        self.enabled = bool_value(self.env.get("DEBUG"), default=False)
        self.output_dir = Path(self.env.get("DEBUG_RESULT_DIR", "debug_result"))
        self.module_path = self.output_dir / "sr_mask_modules.jsonl"
        self.step_path = self.output_dir / "sr_mask_steps.jsonl"
        self.manifest_path = self.output_dir / "sr_mask_manifest.json"
        self.pending_modules: list[dict[str, Any]] = []
        if self.enabled:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.write_json(
                self.manifest_path,
                {
                    "method": self.method_name,
                    "created_at": strftime("%Y%m%d_%H%M%S"),
                    "files": {
                        "modules": str(self.module_path),
                        "steps": str(self.step_path),
                    },
                    "env": {
                        "SR_MASK_MIN": self.env.get("SR_MASK_MIN"),
                        "SR_MASK_MAX": self.env.get("SR_MASK_MAX"),
                        "SR_MASK_EPS": self.env.get("SR_MASK_EPS"),
                        "SR_METRIC_EPS": self.env.get("SR_METRIC_EPS"),
                        "SR_TARGET_LORA_B_CONTAINS": self.env.get("SR_TARGET_LORA_B_CONTAINS"),
                        "SR_MASK_TARGET_LAYERS": self.env.get("SR_MASK_TARGET_LAYERS"),
                    },
                },
            )

    def record_module(
        self,
        *,
        step: int,
        name: str,
        mask: Any,
        base_mask: Any,
        gold_grad: Any,
        hall_grad: Any,
        euclidean_grad: Any,
        adjusted_grad: Any,
        metric_loss: Any,
        gold_progress: Any,
        hall_progress: Any,
        riemannian: Mapping[str, Any] | None = None,
    ) -> None:
        if not self.enabled:
            return

        import torch

        with torch.no_grad():
            m = mask.detach().float()
            bm = base_mask.detach().float()
            g = gold_grad.detach().float()
            h = hall_grad.detach().float()
            eu = euclidean_grad.detach().float()
            adj = adjusted_grad.detach().float()
            hall_masked = h * m
            hall_removed = h * (1.0 - m)

            plain_conflict = torch.relu(-(g * h))
            masked_conflict = torch.relu(-(g * hall_masked))
            plain_conflict_mass = plain_conflict.sum()
            masked_conflict_mass = masked_conflict.sum()
            conflict_reduction = plain_conflict_mass - masked_conflict_mass

            cos_eu_gold = self.cosine(eu, g)
            cos_adj_gold = self.cosine(adj, g)

            row = {
                "step": int(step),
                "method": self.method_name,
                "module": name,
                "mask": self.distribution(m),
                "base_mask": self.distribution(bm),
                "grad_norms": {
                    "gold": self.norm(g),
                    "hall": self.norm(h),
                    "euclidean": self.norm(eu),
                    "adjusted": self.norm(adj),
                    "hall_masked": self.norm(hall_masked),
                    "hall_removed": self.norm(hall_removed),
                },
                "ratios": {
                    "mask_delta_norm_ratio": self.safe_ratio(self.norm(adj - eu), self.norm(eu)),
                    "hall_suppression_ratio": self.safe_ratio(self.norm(hall_masked), self.norm(h)),
                    "hall_removed_ratio": self.safe_ratio(self.norm(hall_removed), self.norm(h)),
                    "masked_vs_euclidean_norm_ratio": self.safe_ratio(self.norm(adj), self.norm(eu)),
                },
                "cosines": {
                    "adjusted_vs_euclidean": self.cosine(adj, eu),
                    "adjusted_vs_gold": cos_adj_gold,
                    "euclidean_vs_gold": cos_eu_gold,
                    "hall_vs_gold": self.cosine(h, g),
                    "hall_masked_vs_gold": self.cosine(hall_masked, g),
                },
                "alignment": {
                    "gold_alignment_gain": cos_adj_gold - cos_eu_gold,
                    "gold_progress_cos": self.scalar(gold_progress),
                    "hall_progress_cos": self.scalar(hall_progress),
                },
                "conflict": {
                    "coord_fraction": self.scalar((g * h < 0).float().mean()),
                    "plain_mass": self.scalar(plain_conflict_mass),
                    "masked_mass": self.scalar(masked_conflict_mass),
                    "reduction_mass": self.scalar(conflict_reduction),
                    "reduction_ratio": self.safe_ratio(self.scalar(conflict_reduction), self.scalar(plain_conflict_mass)),
                    "masked_mass_ratio": self.safe_ratio(self.scalar(masked_conflict_mass), self.scalar(plain_conflict_mass)),
                    "plain_mean": self.scalar(plain_conflict.mean()),
                    "masked_mean": self.scalar(masked_conflict.mean()),
                },
                "losses": {
                    "metric_loss": self.scalar(metric_loss),
                },
            }
            if riemannian is not None:
                row["riemannian"] = {
                    "conflict_energy_plain": self.scalar(riemannian["conflict_energy_plain"]),
                    "conflict_energy_masked": self.scalar(riemannian["conflict_energy_masked"]),
                    "conflict_energy_reduction": self.scalar(riemannian["conflict_energy_reduction"]),
                    "distortion": self.scalar(riemannian["distortion"]),
                    "angle_preservation": self.scalar(riemannian["angle_preservation"]),
                    "geo_gain": self.scalar(riemannian["geo_gain"]),
                    "observed_metric_available": bool(riemannian["observed_metric_available"]),
                    "observed_metric_mean": self.scalar(riemannian["observed_metric_mean"]),
                    "observed_metric_min": self.scalar(riemannian["observed_metric_min"]),
                    "observed_metric_max": self.scalar(riemannian["observed_metric_max"]),
                    "observed_metric_condition": self.scalar(riemannian["observed_metric_condition"]),
                }

        self.pending_modules.append(row)
        self.append_jsonl(self.module_path, row)

    def flush_step(self, step: int) -> None:
        if not self.enabled:
            return
        rows = list(self.pending_modules)
        self.pending_modules.clear()
        if not rows:
            return
        summary = {
            "step": int(step),
            "method": self.method_name,
            "modules": len(rows),
            "microbatch_steps": sorted({int(row["step"]) for row in rows}),
            "module_names": [row["module"] for row in rows],
            "mask_p05_mean": self.mean(row["mask"]["p05"] for row in rows),
            "mask_p50_mean": self.mean(row["mask"]["p50"] for row in rows),
            "mask_p95_mean": self.mean(row["mask"]["p95"] for row in rows),
            "avg_hall_removed_ratio": self.mean(row["ratios"]["hall_removed_ratio"] for row in rows),
            "max_hall_removed_ratio": max(row["ratios"]["hall_removed_ratio"] for row in rows),
            "avg_hall_suppression_ratio": self.mean(row["ratios"]["hall_suppression_ratio"] for row in rows),
            "avg_gold_alignment_gain": self.mean(row["alignment"]["gold_alignment_gain"] for row in rows),
            "avg_conflict_reduction_ratio": self.mean(row["conflict"]["reduction_ratio"] for row in rows),
            "avg_conflict_masked_mass_ratio": self.mean(row["conflict"]["masked_mass_ratio"] for row in rows),
        }
        riemannian_rows = [row["riemannian"] for row in rows if "riemannian" in row]
        if riemannian_rows:
            summary["riemannian"] = {
                "avg_conflict_energy_plain": self.mean(row["conflict_energy_plain"] for row in riemannian_rows),
                "avg_conflict_energy_masked": self.mean(row["conflict_energy_masked"] for row in riemannian_rows),
                "avg_conflict_energy_reduction": self.mean(row["conflict_energy_reduction"] for row in riemannian_rows),
                "avg_distortion": self.mean(row["distortion"] for row in riemannian_rows),
                "avg_angle_preservation": self.mean(row["angle_preservation"] for row in riemannian_rows),
                "avg_geo_gain": self.mean(row["geo_gain"] for row in riemannian_rows),
                "observed_metric_available_ratio": self.mean(1.0 if row["observed_metric_available"] else 0.0 for row in riemannian_rows),
                "avg_observed_metric_mean": self.mean(row["observed_metric_mean"] for row in riemannian_rows),
                "avg_observed_metric_min": self.mean(row["observed_metric_min"] for row in riemannian_rows),
                "avg_observed_metric_max": self.mean(row["observed_metric_max"] for row in riemannian_rows),
                "max_observed_metric_condition": max(row["observed_metric_condition"] for row in riemannian_rows),
            }
        self.append_jsonl(self.step_path, summary)

    @staticmethod
    def distribution(tensor: Any) -> dict[str, Any]:
        import torch

        flat = tensor.detach().float().reshape(-1)
        if flat.numel() == 0:
            return {}
        qs = torch.quantile(flat.cpu(), torch.tensor([0.01, 0.05, 0.5, 0.95, 0.99]))
        return {
            "shape": list(tensor.shape),
            "mean": float(flat.mean().cpu()),
            "std": float(flat.std(unbiased=False).cpu()),
            "min": float(flat.min().cpu()),
            "max": float(flat.max().cpu()),
            "p01": float(qs[0]),
            "p05": float(qs[1]),
            "p50": float(qs[2]),
            "p95": float(qs[3]),
            "p99": float(qs[4]),
            "below_0_99": float((flat < 0.99).float().mean().cpu()),
            "below_0_95": float((flat < 0.95).float().mean().cpu()),
            "below_0_90": float((flat < 0.90).float().mean().cpu()),
            "below_0_75": float((flat < 0.75).float().mean().cpu()),
            "below_0_50": float((flat < 0.50).float().mean().cpu()),
        }

    @staticmethod
    def cosine(a: Any, b: Any) -> float:
        import torch

        af = a.reshape(-1).float()
        bf = b.reshape(-1).float()
        denom = af.norm().clamp_min(1e-12) * bf.norm().clamp_min(1e-12)
        return float((torch.sum(af * bf) / denom).detach().cpu())

    @staticmethod
    def norm(tensor: Any) -> float:
        return float(tensor.detach().float().norm().cpu())

    @staticmethod
    def scalar(value: Any) -> float:
        try:
            return float(value.detach().float().cpu())
        except AttributeError:
            return float(value)

    @staticmethod
    def safe_ratio(num: float, den: float) -> float:
        return float(num) / max(float(den), 1e-12)

    @staticmethod
    def mean(values: Any) -> float:
        vals = [float(value) for value in values]
        return sum(vals) / len(vals) if vals else 0.0

    @staticmethod
    def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    @staticmethod
    def write_json(path: Path, data: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


class SrLoraDebugger:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.env = load_env(args.env_path)
        self.output_dir = self.build_output_dir()
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def build_output_dir(self) -> Path:
        root = Path(self.env.get("OUTPUT_ROOT", "outputs")) / "sr_lora" / "sr_debug"
        return root / strftime("%Y%m%d_%H%M%S")

    def run(self) -> None:
        samples = self.load_samples()
        adapter_dir = Path(self.args.sr_adapter_dir or Path(self.env.get("OUTPUT_ROOT", "outputs")) / "sr_lora" / "adapter")

        adapter_stats = self.adapter_stats(adapter_dir)
        self.write_json("adapter_stats.json", adapter_stats)

        sr_repr = self.collect_representations(samples, adapter_dir)
        sr_points = self.pca2(sr_repr["embeddings"])
        self.write_projection("sr_representation_space.csv", sr_points, samples, label="sr_lora")
        self.plot_projection("sr_representation_space.png", sr_points, ["sr_lora"] * len(sr_points))
        self.write_json(
            "representation_metrics.json",
            {
                "samples": len(samples),
                "sr_adapter_dir": str(adapter_dir),
                "embedding_dim": len(sr_repr["embeddings"][0]) if sr_repr["embeddings"] else 0,
                "pairwise": self.pairwise_metrics(sr_repr["embeddings"]),
            },
        )

        shift_metrics = None
        if self.args.baseline_adapter_dir:
            baseline_dir = Path(self.args.baseline_adapter_dir)
            baseline_repr = self.collect_representations(samples, baseline_dir)
            shift_metrics = self.shift_metrics(baseline_repr["embeddings"], sr_repr["embeddings"])
            self.write_json("baseline_vs_sr_shift.json", shift_metrics)
            self.write_shift_csv("baseline_vs_sr_shift.csv", shift_metrics["per_sample"], samples)
            combined_points = self.pca2(baseline_repr["embeddings"] + sr_repr["embeddings"])
            labels = ["baseline"] * len(samples) + ["sr_lora"] * len(samples)
            self.write_projection("baseline_vs_sr_projection.csv", combined_points, samples + samples, labels=labels)
            self.plot_projection("baseline_vs_sr_projection.png", combined_points, labels)

        self.write_json(
            "manifest.json",
            {
                "output_dir": str(self.output_dir),
                "env_path": self.args.env_path,
                "sr_adapter_dir": str(adapter_dir),
                "baseline_adapter_dir": self.args.baseline_adapter_dir,
                "max_samples": self.args.max_samples,
                "batch_size": self.args.batch_size,
                "files": sorted(path.name for path in self.output_dir.iterdir()),
                "shift_summary": None if shift_metrics is None else shift_metrics["summary"],
            },
        )
        print(f"[sr_debug] saved output_dir={self.output_dir}", flush=True)

    def load_samples(self) -> list[DataPreprocess.Sample]:
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise ImportError("datasets is required for sr_lora debug sampling") from exc

        eval_name = self.env["EVAL_DATASET"]
        if eval_name not in Pipeline.EVAL_SOURCES:
            names = ", ".join(sorted(Pipeline.EVAL_SOURCES))
            raise ValueError(f"EVAL_DATASET must be one of: {names}")

        dataset, subset, split = Pipeline.EVAL_SOURCES[eval_name]
        loaded = load_dataset(dataset, subset, split=split) if subset else load_dataset(dataset, split=split)
        max_samples = self.args.max_samples or optional_int(self.env.get("MAX_EVAL_SAMPLES")) or len(loaded)
        records = [dict(row) for row in loaded.select(range(min(max_samples, len(loaded))))]
        data = DataPreprocess(
            dataset,
            subset,
            split,
            records,
            self.args.env_path,
            sample_limit=max_samples,
            batch_size=self.args.batch_size,
        )
        return list(data.samples)

    def collect_representations(self, samples: Sequence[DataPreprocess.Sample], adapter_dir: Path) -> dict[str, Any]:
        import torch
        from peft import PeftModel

        patch_torch_custom_op_annotations()
        loader = ModelLoad(self.args.env_path)
        tokenizer = loader.load_tokenizer()
        model = loader.load_model()
        if adapter_dir.exists():
            model = PeftModel.from_pretrained(model, str(adapter_dir))
        else:
            print(f"[sr_debug] adapter_dir missing, using base model only: {adapter_dir}", flush=True)
        model.eval()

        embeddings = []
        prompts = [EvalAnswerJudge.base_prompt(sample) for sample in samples]
        for start in range(0, len(prompts), self.args.batch_size):
            chunk = prompts[start : start + self.args.batch_size]
            inputs = tokenizer(
                chunk,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.args.max_prompt_length,
            )
            inputs = {key: value.to(model.device) for key, value in inputs.items()}
            with torch.no_grad():
                out = model(**inputs, output_hidden_states=True, use_cache=False)
            hidden = out.hidden_states[-1]
            mask = inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
            embeddings.extend(pooled.detach().float().cpu().tolist())

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return {"adapter_dir": str(adapter_dir), "embeddings": embeddings}

    def adapter_stats(self, adapter_dir: Path) -> dict[str, Any]:
        state = self.load_adapter_state(adapter_dir)
        tensors = {}
        for key, value in state.items():
            if any(name in key for name in ("lora_A", "lora_B", "sr_u")):
                tensors[key] = self.tensor_stats(value)

        mask_stats = {}
        for key, value in state.items():
            if not key.endswith(".sr_u"):
                continue
            weight_key = f"{key[:-len('.sr_u')]}.weight"
            if weight_key not in state:
                continue
            mask = self.metric_safe_mask(value, state[weight_key])
            mask_stats[key] = self.tensor_stats(mask)

        return {
            "adapter_dir": str(adapter_dir),
            "state_file": str(self.find_adapter_state_file(adapter_dir)),
            "tensor_count": len(state),
            "tracked_tensors": tensors,
            "sr_mask_stats": mask_stats,
        }

    def load_adapter_state(self, adapter_dir: Path) -> Mapping[str, Any]:
        state_file = self.find_adapter_state_file(adapter_dir)
        if state_file is None:
            return {}
        if state_file.suffix == ".safetensors":
            from safetensors.torch import load_file

            return load_file(str(state_file))

        import torch

        return torch.load(state_file, map_location="cpu")

    @staticmethod
    def find_adapter_state_file(adapter_dir: Path) -> Path | None:
        for name in ("adapter_model.safetensors", "adapter_model.bin", "pytorch_model.bin"):
            path = adapter_dir / name
            if path.exists():
                return path
        return None

    def metric_safe_mask(self, u: Any, b: Any) -> Any:
        x = self.normalized(u * b)
        eps = float(self.env["SR_MASK_EPS"])
        threshold = float(self.env["SR_MASK_THRESHOLD"])
        temperature = float(self.env["SR_MASK_TEMPERATURE"])
        return eps + (1.0 - eps) * ((x - threshold) / temperature).sigmoid()

    @staticmethod
    def normalized(x: Any) -> Any:
        return (x - x.mean()) / (x.std(unbiased=False) + 1e-6)

    @staticmethod
    def tensor_stats(tensor: Any) -> dict[str, float | list[int]]:
        data = tensor.detach().float().cpu()
        return {
            "shape": list(data.shape),
            "mean": float(data.mean()),
            "std": float(data.std(unbiased=False)),
            "min": float(data.min()),
            "max": float(data.max()),
            "l2": float(data.norm()),
            "abs_mean": float(data.abs().mean()),
        }

    @staticmethod
    def pca2(embeddings: Sequence[Sequence[float]]) -> list[tuple[float, float]]:
        import numpy as np

        if not embeddings:
            return []
        x = np.asarray(embeddings, dtype=np.float64)
        x = x - x.mean(axis=0, keepdims=True)
        _, _, vt = np.linalg.svd(x, full_matrices=False)
        width = min(2, vt.shape[0])
        coords = x @ vt[:width].T
        if width == 1:
            coords = np.concatenate([coords, np.zeros((coords.shape[0], 1))], axis=1)
        return [(float(row[0]), float(row[1])) for row in coords]

    @staticmethod
    def pairwise_metrics(embeddings: Sequence[Sequence[float]]) -> dict[str, float]:
        import numpy as np

        if len(embeddings) < 2:
            return {"mean_cosine": 0.0, "mean_l2": 0.0}
        x = np.asarray(embeddings, dtype=np.float64)
        norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
        unit = x / norms
        cosine = unit @ unit.T
        l2 = np.linalg.norm(x[:, None, :] - x[None, :, :], axis=-1)
        tri = np.triu_indices(len(x), k=1)
        return {
            "mean_cosine": float(cosine[tri].mean()),
            "mean_l2": float(l2[tri].mean()),
        }

    @staticmethod
    def shift_metrics(baseline: Sequence[Sequence[float]], sr: Sequence[Sequence[float]]) -> dict[str, Any]:
        import numpy as np

        b = np.asarray(baseline, dtype=np.float64)
        s = np.asarray(sr, dtype=np.float64)
        delta = s - b
        l2 = np.linalg.norm(delta, axis=1)
        b_norm = np.linalg.norm(b, axis=1) + 1e-12
        s_norm = np.linalg.norm(s, axis=1) + 1e-12
        cosine = (b * s).sum(axis=1) / (b_norm * s_norm)
        per_sample = [
            {"index": int(i), "l2_shift": float(l2[i]), "cosine": float(cosine[i])}
            for i in range(len(l2))
        ]
        return {
            "summary": {
                "samples": int(len(l2)),
                "mean_l2_shift": float(l2.mean()) if len(l2) else 0.0,
                "max_l2_shift": float(l2.max()) if len(l2) else 0.0,
                "mean_cosine": float(cosine.mean()) if len(cosine) else 0.0,
                "min_cosine": float(cosine.min()) if len(cosine) else 0.0,
            },
            "per_sample": per_sample,
        }

    def write_projection(
        self,
        filename: str,
        points: Sequence[tuple[float, float]],
        samples: Sequence[DataPreprocess.Sample],
        label: str | None = None,
        labels: Sequence[str] | None = None,
    ) -> None:
        path = self.output_dir / filename
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["index", "label", "x", "y", "question", "answer"])
            writer.writeheader()
            for i, (point, sample) in enumerate(zip(points, samples)):
                writer.writerow(
                    {
                        "index": i,
                        "label": labels[i] if labels is not None else label,
                        "x": point[0],
                        "y": point[1],
                        "question": sample.question,
                        "answer": sample.answer,
                    }
                )

    def write_shift_csv(
        self,
        filename: str,
        rows: Sequence[Mapping[str, Any]],
        samples: Sequence[DataPreprocess.Sample],
    ) -> None:
        path = self.output_dir / filename
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["index", "l2_shift", "cosine", "question", "answer"])
            writer.writeheader()
            for row, sample in zip(rows, samples):
                writer.writerow({**row, "question": sample.question, "answer": sample.answer})

    def plot_projection(self, filename: str, points: Sequence[tuple[float, float]], labels: Sequence[str]) -> None:
        if not points:
            return
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            self.write_json(
                f"{Path(filename).stem}_plot_skipped.json",
                {"reason": "matplotlib is not installed", "target": filename},
            )
            return

        colors = {"baseline": "#4C78A8", "sr_lora": "#F58518"}
        plt.figure(figsize=(8, 6))
        for label in sorted(set(labels)):
            xs = [point[0] for point, item_label in zip(points, labels) if item_label == label]
            ys = [point[1] for point, item_label in zip(points, labels) if item_label == label]
            plt.scatter(xs, ys, s=18, alpha=0.75, label=label, color=colors.get(label))
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=180)
        plt.close()

    def write_json(self, filename: str, data: Mapping[str, Any]) -> None:
        (self.output_dir / filename).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze SR-LoRA adapter and representation-space effects.")
    parser.add_argument("--env-path", default="env.sh")
    parser.add_argument("--sr-adapter-dir", default=None)
    parser.add_argument("--baseline-adapter-dir", default=None)
    parser.add_argument("--max-samples", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-prompt-length", type=int, default=4096)
    return parser.parse_args()


def main() -> None:
    SrLoraDebugger(parse_args()).run()


if __name__ == "__main__":
    main()
