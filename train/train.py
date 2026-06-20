from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from pathlib import Path
import sys
from typing import Any

sys.path.append(str(Path(__file__).resolve().parents[1]))

from model.pipeline import Pipeline
from model.eval import Evaluator


class Trainer:
    def __init__(
        self,
        train_records: Sequence[Mapping[str, Any]],
        eval_records: Sequence[Mapping[str, Any]] | None = None,
        env_path: str = "env.sh",
        method_name: str | None = None,
    ) -> None:
        self.env = Pipeline.data_env(env_path) if hasattr(Pipeline, "data_env") else self.load_env(env_path)
        if method_name is not None:
            self.env["METHOD"] = method_name
        self.pipeline = Pipeline(train_records, eval_records, env_path, method_name=method_name)
        self.epochs = int(self.env["TRAIN_EPOCHS"])
        self.max_steps = self.optional_int(self.env["TRAIN_MAX_STEPS"])
        self.early_stop_steps = self.optional_int(self.env.get("TRAIN_EARLY_STOP_STEPS", "none"))
        self.lr = float(self.env["TRAIN_LR"])
        self.weight_decay = float(self.env["TRAIN_WEIGHT_DECAY"])
        self.grad_accum_steps = int(self.env["TRAIN_GRAD_ACCUM_STEPS"])
        self.experiment_dir = self.pipeline.experiment_dir
        self.output_dir = str(self.experiment_dir / "adapter")
        self.trace_path = str(self.experiment_dir / "train_sample_trace.json")
        self.trace_samples = int(self.env["TRAIN_TRACE_SAMPLES"])
        self.train_records = train_records
        self.eval_records = eval_records
        self.trace: dict[str, Any] = {}
        self.write_initial_trace()

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
    def optional_int(value: str) -> int | None:
        return None if value.lower() in {"", "none", "null"} else int(value)

    def train(self, add_hallucinations: bool = False) -> None:
        import torch

        self.reset_training_rng()
        print("[train] loading tokenizer", flush=True)
        tokenizer = self.pipeline.load_tokenizer()
        print("[train] loading model and method adapter", flush=True)
        model = self.pipeline.load_model()
        self.print_trainable(model)
        print(f"[train] model_device={self.model_device(model)} model_dtype={self.model_dtype(model)}", flush=True)
        print(f"[train] method={self.pipeline.method.name}", flush=True)
        loss_fn = self.pipeline.method.loss(tokenizer=tokenizer, max_length=self.pipeline.data.max_length)
        if self.method_needs_hallucinations():
            self.update_trace("first_batch", "created inside each training step after hallucination judge")
        else:
            self.trace_first_batch(loss_fn)
        optimizer = torch.optim.AdamW(
            [param for param in model.parameters() if param.requires_grad],
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        step = 0
        optimizer.zero_grad()
        model.train()
        print(f"[train] start epochs={self.epochs} max_steps={self.max_steps}", flush=True)
        for epoch in range(self.epochs):
            print(f"[train] epoch={epoch + 1}/{self.epochs}", flush=True)
            for batch in self.pipeline.data.batches():
                step += 1
                judge_rows = []
                if self.should_add_hallucinations(add_hallucinations):
                    print(f"[train] step={step} hallucination judge start batch_size={len(batch.samples)}", flush=True)
                    batch, judge_rows = self.pipeline.judged_hallucination_batch(batch)
                    print(
                        f"[train] step={step} hallucination judge done kept={len(batch.samples)} "
                        f"dropped={len(judge_rows) - len(batch.samples)}",
                        flush=True,
                    )
                    if not batch.samples:
                        print(f"[train] step={step} skipped no hallucinated samples", flush=True)
                        self.append_skipped_step_trace(step, judge_rows)
                        self.run_eval_if_needed(model, tokenizer, loss_fn, step)
                        if self.max_steps is not None and step >= self.max_steps:
                            print(f"[train] reached max_steps={self.max_steps}", flush=True)
                            self.flush_optimizer(model, optimizer, step)
                            self.save(model, tokenizer)
                            return
                        if self.should_early_stop(step):
                            self.stop_early(model, tokenizer, optimizer, step)
                            return
                        continue

                if self.method_needs_hallucinations():
                    hall_samples = tuple(
                        sample for sample in batch.samples if getattr(sample, "hallucinated_answer", "")
                    )
                    if not hall_samples:
                        print(f"[train] step={step} skipped no hallucinated_answer", flush=True)
                        self.append_skipped_step_trace(step, judge_rows)
                        self.run_eval_if_needed(model, tokenizer, loss_fn, step)
                        if self.max_steps is not None and step >= self.max_steps:
                            print(f"[train] reached max_steps={self.max_steps}", flush=True)
                            self.flush_optimizer(model, optimizer, step)
                            self.save(model, tokenizer)
                            return
                        if self.should_early_stop(step):
                            self.stop_early(model, tokenizer, optimizer, step)
                            return
                        continue
                    if len(hall_samples) != len(batch.samples):
                        batch = type(batch)(hall_samples)

                print(f"[train] step={step} forward start batch_size={len(batch.samples)}", flush=True)
                out = loss_fn(model, batch.samples)
                print(f"[train] step={step} forward done loss={float(out.loss.detach().cpu()):.4f}", flush=True)
                backward_hook = getattr(self.pipeline.method, "backward", None)
                if backward_hook is not None:
                    backward_hook(model, out, self.grad_accum_steps, step)
                else:
                    (out.loss / self.grad_accum_steps).backward()
                print(f"[train] step={step} backward done", flush=True)

                method_update = None
                if step % self.grad_accum_steps == 0:
                    method_update = self.before_optimizer_step(model, step, optimizer)
                    optimizer.step()
                    after_update = self.after_optimizer_step(model, step, optimizer)
                    if after_update is not None:
                        if method_update is None:
                            method_update = after_update
                        else:
                            method_update = dict(method_update)
                            method_update["after_optimizer_step"] = after_update
                    optimizer.zero_grad()

                print(self.log_line(step, out), flush=True)
                if method_update is not None:
                    print(f"[train] step={step} method_update={method_update}", flush=True)
                self.append_step_trace(step, out, batch.raw(), judge_rows, method_update)

                if self.should_early_stop(step):
                    self.stop_early(model, tokenizer, optimizer, step)
                    return

                self.run_eval_if_needed(model, tokenizer, loss_fn, step)

                if self.max_steps is not None and step >= self.max_steps:
                    print(f"[train] reached max_steps={self.max_steps}", flush=True)
                    self.flush_optimizer(model, optimizer, step)
                    self.save(model, tokenizer)
                    return

        self.flush_optimizer(model, optimizer, step)
        self.save(model, tokenizer)

    def compare_eval(self, step: int) -> dict[str, Any] | None:
        if self.pipeline.eval_data is None:
            return None
        return Evaluator(self.pipeline, output_path=str(self.comparison_path(step))).run()

    def run_eval_if_needed(self, model: Any, tokenizer: Any, loss_fn: Any, step: int) -> None:
        if not self.pipeline.should_eval(step):
            return
        was_training = bool(getattr(model, "training", False))
        torch = None
        cpu_rng_state = None
        cuda_rng_states = None
        try:
            import torch as torch_module

            torch = torch_module
            cpu_rng_state = torch.random.get_rng_state()
            if torch.cuda.is_available() and torch.cuda.is_initialized():
                cuda_rng_states = torch.cuda.get_rng_state_all()
        except ImportError:
            pass

        try:
            print(f"[train] eval start step={step}", flush=True)
            # Always force evaluation mode for every eval path.
            # This covers both eval_loss=True and eval_loss=False; without this,
            # comparison generation could run with dropout/training behavior enabled.
            model.eval()
            if self.env.get("EVAL_LOSS", "true").lower() in {"true", "1", "yes"}:
                eval_loss = self.evaluate(model, tokenizer, loss_fn)
                if eval_loss is not None:
                    print(f"step={step} eval_loss={eval_loss:.4f}", flush=True)
            else:
                print(f"[eval_loss] skipped EVAL_LOSS={self.env.get('EVAL_LOSS')}", flush=True)
            model.eval()
            comparison = self.compare_eval(step)
            if comparison is not None:
                print(f"step={step} comparison={comparison['summary']}", flush=True)
        finally:
            if was_training:
                model.train()
            else:
                model.eval()
            if torch is not None and cpu_rng_state is not None:
                torch.random.set_rng_state(cpu_rng_state)
                if cuda_rng_states is not None:
                    torch.cuda.set_rng_state_all(cuda_rng_states)

    def comparison_path(self, step: int) -> Path:
        return self.experiment_dir / f"step_{step}" / "comparison.json"

    def should_early_stop(self, step: int) -> bool:
        return self.early_stop_steps is not None and step >= self.early_stop_steps

    def stop_early(self, model: Any, tokenizer: Any, optimizer: Any, step: int) -> None:
        print(f"[train] early stopping at step={step} target={self.early_stop_steps}", flush=True)
        self.flush_optimizer(model, optimizer, step)
        self.save(model, tokenizer)

    def should_add_hallucinations(self, add_hallucinations: bool) -> bool:
        return self.method_needs_hallucinations()

    def method_needs_hallucinations(self) -> bool:
        return bool(getattr(self.pipeline.method, "needs_hallucinations", False))

    def reset_training_rng(self) -> None:
        seed = int(self.env.get("TRAIN_SEED", self.env.get("CONTRASTIVE_SEED", "42")))
        try:
            import random

            random.seed(seed)
        except Exception:
            pass
        try:
            import numpy as np

            np.random.seed(seed)
        except Exception:
            pass
        try:
            import torch

            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except Exception:
            pass
        print(f"[train] rng_seed={seed}", flush=True)

    def before_optimizer_step(self, model: Any, step: int, optimizer: Any | None = None) -> dict[str, Any] | None:
        hook = getattr(self.pipeline.method, "before_optimizer_step", None)
        if hook is None:
            return None
        try:
            return hook(model, step, optimizer)
        except TypeError:
            return hook(model, step)

    def after_optimizer_step(self, model: Any, step: int, optimizer: Any | None = None) -> dict[str, Any] | None:
        hook = getattr(self.pipeline.method, "after_optimizer_step", None)
        if hook is None:
            return None
        try:
            return hook(model, step, optimizer)
        except TypeError:
            return hook(model, step)

    def flush_optimizer(self, model: Any, optimizer: Any, step: int) -> None:
        if step > 0 and step % self.grad_accum_steps != 0:
            method_update = self.before_optimizer_step(model, step, optimizer)
            optimizer.step()
            after_update = self.after_optimizer_step(model, step, optimizer)
            if after_update is not None:
                if method_update is None:
                    method_update = after_update
                else:
                    method_update = dict(method_update)
                    method_update["after_optimizer_step"] = after_update
            if method_update is not None:
                print(f"[train] step={step} method_update={method_update}", flush=True)
            optimizer.zero_grad()

    def evaluate(self, model: Any, tokenizer: Any, loss_fn: Any) -> float | None:
        import torch

        if self.pipeline.eval_data is None:
            return None
        model.eval()
        losses = []
        total = len(self.pipeline.eval_data.samples)
        batch_size = self.pipeline.eval_data.batch_size
        print(f"[eval_loss] start samples={total} batch_size={batch_size}", flush=True)
        with torch.no_grad():
            for batch_index, batch in enumerate(self.pipeline.eval_data.batches(), 1):
                start = (batch_index - 1) * batch_size + 1
                end = start + len(batch.samples) - 1
                print(
                    f"[eval_loss] batch={batch_index} samples={start}-{end}/{total} forward start",
                    flush=True,
                )
                batch_loss = float(loss_fn(model, batch.samples).gold_ce.detach().cpu())
                losses.append(batch_loss)
                print(
                    f"[eval_loss] batch={batch_index} samples={start}-{end}/{total} loss={batch_loss:.4f}",
                    flush=True,
                )
        eval_loss = sum(losses) / len(losses) if losses else None
        print(f"[eval_loss] done batches={len(losses)} loss={eval_loss}", flush=True)
        return eval_loss

    def save(self, model: Any, tokenizer: Any) -> None:
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        model.save_pretrained(self.output_dir)
        tokenizer.save_pretrained(self.output_dir)
        print(f"[train] saved output_dir={self.output_dir}", flush=True)

    def write_initial_trace(self) -> None:
        self.trace = {
            "config": {
                "method": self.env.get("METHOD"),
                "needs_hallucinations": bool(getattr(self.pipeline.method, "needs_hallucinations", False)),
                "experiment_dir": str(self.experiment_dir),
                "adapter_dir": self.output_dir,
                "comparison_path_pattern": str(self.experiment_dir / "step_<STEP>" / "comparison.json"),
                "trace_path": self.trace_path,
                "early_stop_steps": self.env.get("TRAIN_EARLY_STOP_STEPS"),
                "model_name": self.env.get("MODEL_NAME"),
                "train_dataset": self.env.get("TRAIN_DATASET"),
                "train_subset": self.env.get("TRAIN_SUBSET"),
                "train_split": self.env.get("TRAIN_SPLIT"),
                "eval_dataset": self.env.get("EVAL_DATASET"),
                "train_batch_size": self.env.get("TRAIN_BATCH_SIZE"),
                "eval_batch_size": self.env.get("EVAL_BATCH_SIZE"),
                "judge_batch_size": self.env.get("JUDGE_BATCH_SIZE"),
                "hallucination_judge_model": self.env.get("HALLUCINATION_JUDGE_MODEL", self.env.get("JUDGE_MODEL")),
                "eval_judge_model": self.env.get("EVAL_JUDGE_MODEL", self.env.get("JUDGE_MODEL")),
                "eval_judge_batch_size": self.env.get("EVAL_JUDGE_BATCH_SIZE", self.env.get("JUDGE_BATCH_SIZE")),
                "max_length": self.env.get("CONTRASTIVE_MAX_LENGTH"),
                "max_train_samples": self.env.get("MAX_TRAIN_SAMPLES"),
                "max_eval_samples": self.env.get("MAX_EVAL_SAMPLES"),
                "sr_target_modules": self.env.get("SR_TARGET_MODULES"),
                "sr_target_layers": self.env.get("SR_TARGET_LAYERS"),
                "sr_lambda": self.env.get("SR_LAMBDA"),
            },
            "raw_train_records": [dict(row) for row in self.train_records[: self.trace_samples]],
            "normalized_train_samples": self.sample_preview(self.pipeline.data.samples),
            "eval_samples": self.sample_preview(self.pipeline.eval_data.samples) if self.pipeline.eval_data else [],
            "steps": [],
        }
        self.write_trace()

    def trace_first_batch(self, loss_fn: Any) -> None:
        first_batch = next(self.pipeline.data.batches(), None)
        if first_batch is None:
            self.update_trace("first_batch", None)
            return
        summary = {
            "raw_batch": first_batch.raw(),
            "gold_lm_batch": self.batch_summary(loss_fn, first_batch.samples, "answer"),
        }
        hall_samples = [sample for sample in first_batch.samples if getattr(sample, "hallucinated_answer", "")]
        if hall_samples:
            summary["hall_lm_batch"] = self.batch_summary(loss_fn, hall_samples, "hallucinated_answer")
        else:
            summary["hall_lm_batch"] = "no hallucinated_answer in first batch"
        self.update_trace("first_batch", summary)

    def batch_summary(self, loss_fn: Any, samples: Sequence[Any], answer_attr: str) -> list[dict[str, Any]]:
        rows = []
        for sample in samples[: self.trace_samples]:
            encoded = loss_fn.encode_sample(sample, getattr(sample, answer_attr))
            rows.append(
                {
                    "question": sample.question,
                    "reference_preview": sample.reference[:200],
                    "answer_attr": answer_attr,
                    "answer": getattr(sample, answer_attr),
                    "input_tokens": len(encoded["input_ids"]),
                    "loss_tokens": sum(label != -100 for label in encoded["labels"]),
                    "ignored_tokens": sum(label == -100 for label in encoded["labels"]),
                }
            )
        return rows

    def sample_preview(self, samples: Sequence[Any]) -> list[dict[str, str]]:
        return [sample.as_dict() for sample in samples[: self.trace_samples]]

    def append_step_trace(
        self,
        step: int,
        out: Any,
        raw_batch: list[dict[str, str]],
        judge_rows: Sequence[Mapping[str, Any]] | None = None,
        method_update: Mapping[str, Any] | None = None,
    ) -> None:
        row = {
            "step": step,
            "loss": float(out.loss.detach().cpu()),
            "gold_ce": float(out.gold_ce.detach().cpu()),
            "hall_ce": None if out.hall_ce is None else float(out.hall_ce.detach().cpu()),
            "raw_batch": raw_batch[: self.trace_samples],
        }
        if judge_rows is not None:
            row["hallucination_judge"] = list(judge_rows[: self.trace_samples])
        if method_update is not None:
            row["method_update"] = dict(method_update)
        self.trace["steps"].append(row)
        self.write_trace()

    def append_skipped_step_trace(self, step: int, judge_rows: Sequence[Mapping[str, Any]]) -> None:
        self.trace["steps"].append(
            {
                "step": step,
                "skipped": True,
                "reason": "no hallucinated samples in judged batch",
                "hallucination_judge": list(judge_rows[: self.trace_samples]),
            }
        )
        self.write_trace()

    def update_trace(self, key: str, value: Any) -> None:
        self.trace[key] = value
        self.write_trace()

    def write_trace(self) -> None:
        path = Path(self.trace_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.trace, indent=2, ensure_ascii=False), encoding="utf-8")

    @staticmethod
    def print_trainable(model: Any) -> None:
        if hasattr(model, "print_trainable_parameters"):
            model.print_trainable_parameters()
            return
        total = sum(param.numel() for param in model.parameters())
        trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
        print(f"[train] trainable_parameters={trainable}/{total}", flush=True)

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
    def log_line(step: int, out: Any) -> str:
        gold = float(out.gold_ce.detach().cpu())
        hall = None if out.hall_ce is None else float(out.hall_ce.detach().cpu())
        total = float(out.loss.detach().cpu())
        return f"step={step} loss={total:.4f} gold_ce={gold:.4f} hall_ce={hall}"


if __name__ == "__main__":
    records = [
        {
            "knowledge": "LoRA freezes the base model and trains small adapter weights.",
            "question": "What does LoRA train?",
            "right_answer": "LoRA trains small adapter weights.",
        }
    ]
    trainer = Trainer(records)
    print(
        {
            "train_samples": len(trainer.pipeline.data.samples),
            "epochs": trainer.epochs,
            "lr": trainer.lr,
            "output_dir": trainer.output_dir,
        }
    )

