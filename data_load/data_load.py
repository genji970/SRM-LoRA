from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pprint import pprint
from typing import Any


class DataLoad:
    @dataclass(frozen=True)
    class Source:
        dataset: str
        subset: str
        split: str

    @dataclass(frozen=True)
    class Sample:
        question: str
        reference: str
        answer: str

        def as_dict(self) -> dict[str, str]:
            return {"question": self.question, "reference": self.reference, "answer": self.answer}

    TRAIN_SOURCES = (Source("pminervini/HaluEval", "qa", "data"),)
    EVAL_SOURCES = (
        Source("pminervini/HaluEval", "summarization", "data"),
        Source("pminervini/HaluEval", "dialogue", "data"),
        Source("ucinlp/drop", "", "validation"),
        Source("hotpotqa/hotpot_qa", "fullwiki", "validation"),
    )

    def __init__(self, dataset: str, subset: str, split: str, records: Sequence[Mapping[str, Any]]) -> None:
        self.source = self.Source(dataset, subset, split)
        self.samples = self.normalize(self.source, records)

    def as_list(self) -> list[dict[str, str]]:
        return [sample.as_dict() for sample in self.samples]

    @classmethod
    def normalize(cls, source: Source, records: Sequence[Mapping[str, Any]]) -> list[Sample]:
        cls.check_type("records", records, Sequence)
        convert = cls.converter(source)
        return [convert(record) for record in records]

    @classmethod
    def converter(cls, source: Source) -> Any:
        if source.dataset == "pminervini/HaluEval" and source.split == "data":
            return {
                "qa": cls.halueval_qa,
                "dialogue": cls.halueval_dialogue,
                "summarization": cls.halueval_summarization,
            }.get(source.subset) or cls.unsupported(source)
        if source.dataset == "ucinlp/drop" and source.subset == "" and source.split == "validation":
            return cls.drop_validation
        if source.dataset == "hotpotqa/hotpot_qa" and source.subset == "fullwiki" and source.split == "validation":
            return cls.hotpotqa_fullwiki
        return cls.unsupported(source)

    @staticmethod
    def unsupported(source: Source) -> Any:
        raise ValueError(f"unsupported source: {source.dataset} / {source.subset} / {source.split}")

    @classmethod
    def halueval_qa(cls, record: Mapping[str, Any]) -> Sample:
        return cls.Sample(
            question=cls.text(record["question"]),
            reference=cls.text(record["knowledge"]),
            answer=cls.text(record["right_answer"]),
        )

    @classmethod
    def halueval_dialogue(cls, record: Mapping[str, Any]) -> Sample:
        return cls.Sample(
            question=cls.text(record["dialogue_history"]),
            reference=cls.text(record["knowledge"]),
            answer=cls.text(record["right_response"]),
        )

    @classmethod
    def halueval_summarization(cls, record: Mapping[str, Any]) -> Sample:
        return cls.Sample(
            question="",
            reference=cls.text(record["document"]),
            answer=cls.text(record["right_summary"]),
        )

    @classmethod
    def drop_validation(cls, record: Mapping[str, Any]) -> Sample:
        spans = record["answers_spans"]["spans"]
        cls.check_type("answers_spans.spans", spans, Sequence)
        if not spans:
            raise ValueError("answers_spans.spans must contain at least one answer")
        return cls.Sample(
            question=cls.text(record["question"]),
            reference=cls.text(record["passage"]),
            answer=cls.text(spans[0]),
        )

    @classmethod
    def hotpotqa_fullwiki(cls, record: Mapping[str, Any]) -> Sample:
        context = record["context"]
        cls.check_type("context", context, Mapping)

        titles = context.get("title", [])
        sentences = context.get("sentences", [])
        cls.check_type("context.title", titles, Sequence)
        cls.check_type("context.sentences", sentences, Sequence)

        reference_parts = []
        for title, sentence_list in zip(titles, sentences):
            title_text = cls.text(str(title))
            if isinstance(sentence_list, Sequence) and not isinstance(sentence_list, str):
                passage_text = " ".join(cls.text(str(sentence)) for sentence in sentence_list)
            else:
                passage_text = cls.text(str(sentence_list))
            if title_text and passage_text:
                reference_parts.append(f"{title_text}: {passage_text}")
            elif passage_text:
                reference_parts.append(passage_text)

        return cls.Sample(
            question=cls.text(record["question"]),
            reference=cls.text(" ".join(reference_parts)),
            answer=cls.text(record["answer"]),
        )

    @classmethod
    def text(cls, value: str) -> str:
        cls.check_type("text", value, str)
        return " ".join(value.strip().split())

    @staticmethod
    def check_type(name: str, value: Any, expected: type | tuple[type, ...]) -> None:
        if not isinstance(value, expected):
            names = " | ".join(t.__name__ for t in expected) if isinstance(expected, tuple) else expected.__name__
            raise TypeError(f"{name}: expected {names}, got {type(value).__name__}")


if __name__ == "__main__":
    examples = [
        (
            "pminervini/HaluEval",
            "qa",
            "data",
            [{"knowledge": "LoRA freezes base weights.", "question": "What is frozen?", "right_answer": "Base weights."}],
        ),
        (
            "pminervini/HaluEval",
            "summarization",
            "data",
            [{"document": "A long article.", "right_summary": "A short summary."}],
        ),
        (
            "pminervini/HaluEval",
            "dialogue",
            "data",
            [{"knowledge": "It closes at 8 PM.", "dialogue_history": "When close?", "right_response": "At 8 PM."}],
        ),
        (
            "ucinlp/drop",
            "",
            "validation",
            [{"passage": "It began in 1999.", "question": "When?", "answers_spans": {"spans": ["1999"]}}],
        ),
        (
            "hotpotqa/hotpot_qa",
            "fullwiki",
            "validation",
            [
                {
                    "question": "Who wrote the novel?",
                    "answer": "Alice",
                    "context": {
                        "title": ["Novel", "Author"],
                        "sentences": [["The novel was published in 1999."], ["Alice wrote the novel."]],
                    },
                }
            ],
        ),
    ]
    for dataset, subset, split, records in examples:
        print(f"\n[{dataset} / {subset or '-'} / {split}]")
        pprint(DataLoad(dataset, subset, split, records).as_list(), sort_dicts=False)
