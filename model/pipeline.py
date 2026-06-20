from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path
import re
import sys
from typing import Any

sys.path.append(str(Path(__file__).resolve().parents[1]))

from model.data_preprocess import DataPreprocess
from model.method import METHODS
from model.model_load import ModelLoad, patch_torch_custom_op_annotations



class TrainingHallucinationJudge:
    def __init__(self, env_path: str = "env.sh") -> None:
        env = DataPreprocess.load_env(env_path)
        self.base_model_name = env["FROZEN_BASE_MODEL"]
        self.judge_model_name = env.get("HALLUCINATION_JUDGE_MODEL", env["JUDGE_MODEL"])
        self.base_max_prompt_length = int(env["FROZEN_BASE_MAX_PROMPT_LENGTH"])
        self.base_max_new_tokens = int(env["FROZEN_BASE_MAX_NEW_TOKENS"])
        self.judge_max_prompt_length = int(env.get("HALLUCINATION_JUDGE_MAX_PROMPT_LENGTH", env["JUDGE_MAX_PROMPT_LENGTH"]))
        self.judge_max_new_tokens = int(env.get("HALLUCINATION_JUDGE_MAX_NEW_TOKENS", env["JUDGE_MAX_NEW_TOKENS"]))
        self.judge_batch_size = int(env.get("HALLUCINATION_JUDGE_BATCH_SIZE", env["JUDGE_BATCH_SIZE"]))
        self.loader = ModelLoad(env_path)
        self.model_kwargs = self.loader.model_kwargs()
        self.tokenizer_kwargs = self.loader.tokenizer_kwargs()
        self.base_model: Any | None = None
        self.base_tokenizer: Any | None = None
        self.judge_model: Any | None = None
        self.judge_tokenizer: Any | None = None

    def predict(self, sample: DataPreprocess.Sample) -> str:
        return self.generate_many(
            self.base_model_name,
            [self.base_prompt(sample)],
            self.base_max_prompt_length,
            self.base_max_new_tokens,
            model_attr="base_model",
            tokenizer_attr="base_tokenizer",
        )[0]

    def predict_many(self, samples: Sequence[DataPreprocess.Sample]) -> list[str]:
        if not samples:
            return []
        return self.generate_many(
            self.base_model_name,
            [self.base_prompt(sample) for sample in samples],
            self.base_max_prompt_length,
            self.base_max_new_tokens,
            model_attr="base_model",
            tokenizer_attr="base_tokenizer",
        )

    def judge(self, sample: DataPreprocess.Sample, prediction: str) -> bool:
        if self.contains_gold_answer(sample.answer, prediction):
            return False
        output = self.generate_many(
            self.judge_model_name,
            [self.judge_prompt(sample, prediction)],
            self.judge_max_prompt_length,
            self.judge_max_new_tokens,
            model_attr="judge_model",
            tokenizer_attr="judge_tokenizer",
        )[0]
        return self.parse_judge_output(output)

    def judge_many(self, samples: Sequence[DataPreprocess.Sample], predictions: Sequence[str]) -> list[bool]:
        if not samples:
            return []
        results: list[bool | None] = []
        pending_samples = []
        pending_predictions = []
        pending_indexes = []
        for index, (sample, prediction) in enumerate(zip(samples, predictions)):
            if self.contains_gold_answer(sample.answer, prediction):
                results.append(False)
                continue
            results.append(None)
            pending_samples.append(sample)
            pending_predictions.append(prediction)
            pending_indexes.append(index)
        if not pending_samples:
            return [bool(result) for result in results]
        outputs = self.generate_many(
            self.judge_model_name,
            [self.judge_prompt(sample, prediction) for sample, prediction in zip(pending_samples, pending_predictions)],
            self.judge_max_prompt_length,
            self.judge_max_new_tokens,
            model_attr="judge_model",
            tokenizer_attr="judge_tokenizer",
        )
        for index, output in zip(pending_indexes, outputs):
            results[index] = self.parse_judge_output(output)
        return [bool(result) for result in results]

    def run(self, sample: DataPreprocess.Sample) -> tuple[str, bool]:
        prediction = self.predict(sample)
        return prediction, self.judge(sample, prediction)

    def run_many(self, samples: Sequence[DataPreprocess.Sample]) -> list[tuple[str, bool]]:
        predictions = self.predict_many(samples)
        judgments = self.judge_many(samples, predictions)
        return list(zip(predictions, judgments))

    def generate(
        self,
        model_name: str,
        prompt: str,
        max_prompt_length: int,
        max_new_tokens: int,
        model_attr: str,
        tokenizer_attr: str,
    ) -> str:
        return self.generate_many(
            model_name,
            [prompt],
            max_prompt_length,
            max_new_tokens,
            model_attr,
            tokenizer_attr,
        )[0]

    def generate_many(
        self,
        model_name: str,
        prompts: Sequence[str],
        max_prompt_length: int,
        max_new_tokens: int,
        model_attr: str,
        tokenizer_attr: str,
    ) -> list[str]:
        import torch

        patch_torch_custom_op_annotations()

        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = getattr(self, tokenizer_attr)
        model = getattr(self, model_attr)
        if tokenizer is None:
            tokenizer = self.shared_tokenizer(model_name, tokenizer_attr) or AutoTokenizer.from_pretrained(model_name, **self.tokenizer_kwargs)
            tokenizer.padding_side = "left"
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
            setattr(self, tokenizer_attr, tokenizer)
        if model is None:
            model = self.shared_model(model_name, model_attr) or AutoModelForCausalLM.from_pretrained(
                model_name,
                **self.model_kwargs,
            )
            setattr(self, model_attr, model)
        model.eval()

        inputs = tokenizer(
            list(prompts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_prompt_length,
        )
        inputs = {key: value.to(model.device) for key, value in inputs.items()}
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.pad_token_id)
        prompt_width = inputs["input_ids"].shape[-1]
        return [
            tokenizer.decode(row[prompt_width:], skip_special_tokens=True).strip()
            for row in output_ids
        ]

    def shared_model(self, model_name: str, model_attr: str) -> Any | None:
        if model_attr != "base_model" and model_name == self.base_model_name and self.base_model is not None:
            return self.base_model
        return None

    def shared_tokenizer(self, model_name: str, tokenizer_attr: str) -> Any | None:
        if tokenizer_attr != "base_tokenizer" and model_name == self.base_model_name and self.base_tokenizer is not None:
            return self.base_tokenizer
        return None

    @staticmethod
    def base_prompt(sample: DataPreprocess.Sample) -> str:
        return (
            "Answer the question using the reference.\n\n"
            f"Reference:\n{sample.reference}\n\n"
            f"Question:\n{sample.question}\n\n"
            "Answer:"
        )

    @staticmethod
    def judge_prompt(sample: DataPreprocess.Sample, prediction: str) -> str:
        return (
            "Decide whether the prediction is hallucinated. Answer only yes or no.\n"
            "Mark no when the gold answer is present in the prediction, even if the "
            "prediction also includes extra explanation, punctuation, or surrounding text. "
            "Mark yes only when the prediction contradicts the reference, is unsupported by "
            "the reference, or fails to include the gold answer or a semantically equivalent answer.\n\n"
            f"Reference:\n{sample.reference}\n\n"
            f"Question:\n{sample.question}\n\n"
            f"Gold answer:\n{sample.answer}\n\n"
            f"Prediction:\n{prediction}\n\n"
            "Hallucinated?"
        )

    @staticmethod
    def parse_judge_output(output: str) -> bool:
        text = output.strip().lower()
        if text.startswith(("yes", "true", "hallucinated")):
            return True
        if text.startswith(("no", "false", "not hallucinated", "supported")):
            return False
        return "yes" in text and "no" not in text

    @classmethod
    def contains_gold_answer(cls, answer: str, prediction: str) -> bool:
        normalized_answer = cls.normalize_for_match(answer)
        normalized_prediction = cls.normalize_for_match(prediction)
        if not normalized_answer or not normalized_prediction:
            return False
        return f" {normalized_answer} " in f" {normalized_prediction} "

    @staticmethod
    def normalize_for_match(value: str) -> str:
        return " ".join(re.sub(r"[^0-9a-zA-Z]+", " ", value.lower()).split())

    def clear_models(self) -> None:
        self.base_model = None
        self.base_tokenizer = None
        self.judge_model = None
        self.judge_tokenizer = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


class EvalAnswerJudge(TrainingHallucinationJudge):
    def __init__(self, env_path: str = "env.sh") -> None:
        super().__init__(env_path)
        env = DataPreprocess.load_env(env_path)
        self.answer_judge_model_name = env.get("EVAL_JUDGE_MODEL", env["JUDGE_MODEL"])
        self.answer_judge_max_prompt_length = int(env.get("EVAL_JUDGE_MAX_PROMPT_LENGTH", env["JUDGE_MAX_PROMPT_LENGTH"]))
        self.answer_judge_max_new_tokens = int(env.get("EVAL_JUDGE_MAX_NEW_TOKENS", env["JUDGE_MAX_NEW_TOKENS"]))
        self.judge_batch_size = int(env.get("EVAL_JUDGE_BATCH_SIZE", env["JUDGE_BATCH_SIZE"]))
        self.answer_judge_model: Any | None = None
        self.answer_judge_tokenizer: Any | None = None

    def judge_correct(self, sample: DataPreprocess.Sample, prediction: str) -> bool:
        if self.contains_gold_answer(sample.answer, prediction):
            return True
        output = self.generate_many(
            self.answer_judge_model_name,
            [self.answer_prompt(sample, prediction)],
            self.answer_judge_max_prompt_length,
            self.answer_judge_max_new_tokens,
            model_attr="answer_judge_model",
            tokenizer_attr="answer_judge_tokenizer",
        )[0]
        return self.parse_answer_output(output)

    def judge_correct_many(self, samples: Sequence[DataPreprocess.Sample], predictions: Sequence[str]) -> list[bool]:
        if not samples:
            return []
        results: list[bool | None] = []
        pending_samples = []
        pending_predictions = []
        pending_indexes = []
        for index, (sample, prediction) in enumerate(zip(samples, predictions)):
            if self.contains_gold_answer(sample.answer, prediction):
                results.append(True)
                continue
            results.append(None)
            pending_samples.append(sample)
            pending_predictions.append(prediction)
            pending_indexes.append(index)
        if not pending_samples:
            return [bool(result) for result in results]
        outputs = self.generate_many(
            self.answer_judge_model_name,
            [self.answer_prompt(sample, prediction) for sample, prediction in zip(pending_samples, pending_predictions)],
            self.answer_judge_max_prompt_length,
            self.answer_judge_max_new_tokens,
            model_attr="answer_judge_model",
            tokenizer_attr="answer_judge_tokenizer",
        )
        for index, output in zip(pending_indexes, outputs):
            results[index] = self.parse_answer_output(output)
        return [bool(result) for result in results]

    @staticmethod
    def answer_prompt(sample: DataPreprocess.Sample, prediction: str) -> str:
        return (
            "Decide whether the prediction should be accepted as a correct answer. "
            "Use semantic equivalence, not exact string matching. Answer only yes or no.\n"
            "Accept the prediction when it contains the gold answer or a semantically equivalent "
            "answer anywhere in the text, even if it also includes extra explanation, punctuation, "
            "or surrounding words. Reject only when the answer is absent, contradicted, or unsupported.\n\n"
            f"Reference:\n{sample.reference}\n\n"
            f"Question:\n{sample.question}\n\n"
            f"Gold answer:\n{sample.answer}\n\n"
            f"Prediction:\n{prediction}\n\n"
            "Correct answer?"
        )

    @staticmethod
    def parse_answer_output(output: str) -> bool:
        text = output.strip().lower()
        if text.startswith(("yes", "true", "correct", "acceptable")):
            return True
        if text.startswith(("no", "false", "incorrect", "wrong", "not correct")):
            return False
        return "yes" in text and "no" not in text

    def clear_models(self) -> None:
        super().clear_models()
        self.answer_judge_model = None
        self.answer_judge_tokenizer = None


HallucinationJudge = TrainingHallucinationJudge


class Pipeline:
    @dataclass(frozen=True)
    class ModelBatch:
        raw: list[dict[str, str]]
        inputs: dict[str, Any]

    EVAL_SOURCES = {
        "drop": ("ucinlp/drop", "", "validation"),
        "dialogue": ("pminervini/HaluEval", "dialogue", "data"),
        "summarization": ("pminervini/HaluEval", "summarization", "data"),
        "hotpotqa_fullwiki": ("hotpotqa/hotpot_qa", "fullwiki", "validation"),
    }

    def __init__(
        self,
        train_records: Sequence[Mapping[str, Any]],
        eval_records: Sequence[Mapping[str, Any]] | None = None,
        env_path: str = "env.sh",
        method_name: str | None = None,
    ) -> None:
        env = DataPreprocess.load_env(env_path)
        self.env = env
        self.dataset = env["TRAIN_DATASET"]
        self.subset = env["TRAIN_SUBSET"]
        self.split = env["TRAIN_SPLIT"]
        self.eval_name = env["EVAL_DATASET"]
        self.method_name = method_name or env.get("METHOD", "contrastive_lora")
        self.env["METHOD"] = self.method_name
        self.experiment_dir = Path(env.get("OUTPUT_ROOT", "outputs")) / self.method_name / self.safe_path_name(self.eval_name)
        self.max_train_samples = DataPreprocess.optional_int(env["MAX_TRAIN_SAMPLES"])
        self.max_eval_samples = DataPreprocess.optional_int(env["MAX_EVAL_SAMPLES"])
        self.eval_every_steps = int(env["EVAL_EVERY_STEPS"])
        self.eval_start_step = int(env.get("EVAL_START_STEP", "1"))
        self.train_batch_size = int(env["TRAIN_BATCH_SIZE"])
        self.eval_batch_size = int(env["EVAL_BATCH_SIZE"])
        self.data = DataPreprocess(
            self.dataset,
            self.subset,
            self.split,
            train_records,
            env_path,
            sample_limit=self.max_train_samples,
            batch_size=self.train_batch_size,
        )
        self.eval_data = self.build_eval_data(eval_records, env_path) if eval_records is not None else None
        self.model_loader = ModelLoad(env_path)
        self.method = self.build_method(env_path)
        self.hallucination_judge = TrainingHallucinationJudge(env_path)
        self.eval_answer_judge = EvalAnswerJudge(env_path)
        self.tokenizer: Any | None = None
        self.model: Any | None = None

    def build_method(self, env_path: str) -> Any:
        if self.method_name not in METHODS:
            valid = ", ".join(sorted(METHODS))
            raise ValueError(f"METHOD must be one of: {valid}")
        return METHODS[self.method_name](env_path)

    def load_tokenizer(self) -> Any:
        if self.tokenizer is None:
            self.tokenizer = self.model_loader.load_tokenizer()
        return self.tokenizer

    def load_model(self) -> Any:
        if self.model is None:
            check_dependencies = getattr(self.method, "check_dependencies", None)
            if check_dependencies is not None:
                check_dependencies()
            self.model = self.method.apply(self.model_loader.load_model())
        return self.model

    def model_batches(self, tokenizer: Any | None = None) -> Iterator[ModelBatch]:
        tokenizer = tokenizer or self.load_tokenizer()
        for batch in self.data.batches():
            yield self.ModelBatch(
                raw=batch.raw(),
                inputs=self.data.model_inputs(batch, tokenizer),
            )

    def eval_batches(self, tokenizer: Any | None = None) -> Iterator[ModelBatch]:
        if self.eval_data is None:
            return
        tokenizer = tokenizer or self.load_tokenizer()
        for batch in self.eval_data.batches():
            yield self.ModelBatch(
                raw=batch.raw(),
                inputs=self.eval_data.model_inputs(batch, tokenizer),
            )

    def should_eval(self, step: int) -> bool:
        return self.eval_every_steps > 0 and step >= self.eval_start_step and step % self.eval_every_steps == 0

    def build_eval_data(self, records: Sequence[Mapping[str, Any]], env_path: str) -> DataPreprocess:
        if self.eval_name not in self.EVAL_SOURCES:
            names = ", ".join(sorted(self.EVAL_SOURCES))
            raise ValueError(f"EVAL_DATASET must be one of: {names}")
        dataset, subset, split = self.EVAL_SOURCES[self.eval_name]
        return DataPreprocess(
            dataset,
            subset,
            split,
            records,
            env_path,
            sample_limit=self.max_eval_samples,
            batch_size=self.eval_batch_size,
        )

    @staticmethod
    def safe_path_name(value: str) -> str:
        cleaned = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value.strip())
        return cleaned or "eval"

    def judged_hallucination_batch(
        self,
        batch: DataPreprocess.Batch,
        judge: TrainingHallucinationJudge | None = None,
    ) -> tuple[DataPreprocess.Batch, list[dict[str, Any]]]:
        judge = judge or self.hallucination_judge
        kept = []
        rows = []
        results = judge.run_many(batch.samples)
        for sample, (prediction, is_hallucinated) in zip(batch.samples, results):
            row = {
                "question": sample.question,
                "reference_preview": sample.reference[:200],
                "answer": sample.answer,
                "prediction": prediction,
                "is_hallucinated": is_hallucinated,
            }
            rows.append(row)
            if is_hallucinated:
                kept.append(
                    replace(
                        sample,
                        hallucinated_answer=DataPreprocess.text(prediction),
                    )
                )
        return DataPreprocess.Batch(tuple(kept)), rows

    def clear_judge_models(self) -> None:
        self.hallucination_judge.clear_models()
        self.eval_answer_judge.clear_models()


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

    pipeline = Pipeline(records)
    for batch in pipeline.data.batches():
        print(batch.raw())
