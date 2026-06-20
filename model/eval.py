from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any

sys.path.append(str(Path(__file__).resolve().parents[1]))

from model.data_preprocess import DataPreprocess
from model.pipeline import EvalAnswerJudge, Pipeline


class Evaluator:
    @dataclass(frozen=True)
    class Row:
        question: str
        reference: str
        answer: str
        frozen_prediction: str
        method_prediction: str
        frozen_correct: bool
        method_correct: bool

        def as_dict(self) -> dict[str, Any]:
            return {
                "question": self.question,
                "reference": self.reference,
                "answer": self.answer,
                "frozen_prediction": self.frozen_prediction,
                "method_prediction": self.method_prediction,
                "frozen_correct": self.frozen_correct,
                "method_correct": self.method_correct,
            }

    def __init__(self, pipeline: Pipeline, output_path: str | None = None) -> None:
        if pipeline.eval_data is None:
            raise ValueError("pipeline.eval_data is missing")
        self.pipeline = pipeline
        self.output_path = Path(output_path) if output_path else pipeline.experiment_dir / "comparison.json"
        self.judge = pipeline.eval_answer_judge

    def run(self) -> dict[str, Any]:
        samples = self.pipeline.eval_data.samples
        rows = []
        for start in range(0, len(samples), self.judge.judge_batch_size):
            chunk = samples[start : start + self.judge.judge_batch_size]
            print(f"[eval] answer judge samples {start + 1}-{start + len(chunk)} / {len(samples)}", flush=True)
            rows.extend(self.evaluate_batch(chunk))
        summary = self.summary(rows)
        samples_path = self.save_samples(rows)
        result = {
            "summary": summary,
            "samples_path": str(samples_path),
            "samples_format": "jsonl",
        }
        self.save(result)
        return result

    def evaluate_sample(self, sample: DataPreprocess.Sample) -> Row:
        return self.evaluate_batch([sample])[0]

    def evaluate_batch(self, samples: Sequence[DataPreprocess.Sample]) -> list[Row]:
        frozen_predictions = self.judge.predict_many(samples)
        frozen_corrects = self.judge.judge_correct_many(samples, frozen_predictions)
        method_predictions = self.method_predict_many(samples)
        method_corrects = self.judge.judge_correct_many(samples, method_predictions)
        return [
            self.Row(
                question=sample.question,
                reference=sample.reference,
                answer=sample.answer,
                frozen_prediction=frozen_prediction,
                method_prediction=method_prediction,
                frozen_correct=frozen_correct,
                method_correct=method_correct,
            )
            for sample, frozen_prediction, method_prediction, frozen_correct, method_correct in zip(
                samples,
                frozen_predictions,
                method_predictions,
                frozen_corrects,
                method_corrects,
            )
        ]

    def method_predict(self, sample: DataPreprocess.Sample) -> str:
        return self.method_predict_many([sample])[0]

    def method_predict_many(self, samples: Sequence[DataPreprocess.Sample]) -> list[str]:
        if not samples:
            return []
        model = self.pipeline.load_model()
        model.eval()
        tokenizer = self.pipeline.load_tokenizer()
        prompts = [EvalAnswerJudge.base_prompt(sample) for sample in samples]
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.judge.base_max_prompt_length,
        )
        inputs = {key: value.to(model.device) for key, value in inputs.items()}

        import torch

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=self.judge.base_max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
            )
        prompt_width = inputs["input_ids"].shape[-1]
        return [
            tokenizer.decode(row[prompt_width:], skip_special_tokens=True).strip()
            for row in output_ids
        ]

    @staticmethod
    def summary(rows: Sequence[Row]) -> dict[str, int | float]:
        improved = sum((not row.frozen_correct) and row.method_correct for row in rows)
        worsened = sum(row.frozen_correct and not row.method_correct for row in rows)
        frozen_correct = sum(row.frozen_correct for row in rows)
        method_correct = sum(row.method_correct for row in rows)
        evaluated = len(rows)
        frozen_incorrect = evaluated - frozen_correct
        method_incorrect = evaluated - method_correct
        return {
            "evaluated": evaluated,
            "frozen_base_correct": frozen_correct,
            "method_correct": method_correct,
            "frozen_base_incorrect": frozen_incorrect,
            "method_incorrect": method_incorrect,
            "frozen_base_hallucination_rate": frozen_incorrect / evaluated if evaluated else 0.0,
            "method_hallucination_rate": method_incorrect / evaluated if evaluated else 0.0,
            "improved": improved,
            "worsened": worsened,
            "net_improved": improved - worsened,
        }

    def save(self, result: Mapping[str, Any]) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(
            json.dumps(result, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def save_samples(self, rows: Sequence[Row]) -> Path:
        samples_path = self.output_path.with_name(f"{self.output_path.stem}_samples.jsonl")
        samples_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [json.dumps(row.as_dict(), ensure_ascii=False) for row in rows]
        samples_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        return samples_path


if __name__ == "__main__":
    print({"output_path": "outputs/<METHOD>/<EVAL_DATASET>/step_<STEP>/comparison.json"})
