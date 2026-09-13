"""
FARM dataset schema normalization.

Why this exists as its own module
----------------------------------
The Farm dataset (Xu et al., ACL 2024) ships 4 subsets with genuinely
different raw JSON schemas:

  - Boolq.jsonl      : {"question", "answer": bool, "source", "adv": {...}}
  - NQ1.jsonl        : {"question", "answer": str,  "source", "adv": {..., "mcq"}}
  - NQ2.jsonl        : same shape as NQ1 (differs only in how "adv.target"
                        was chosen upstream — semantically distinct, not
                        structurally distinct)
  - TruthfulQA.jsonl : {"type", "category", "question", "best_answer",
                        "correct_answer", "incorrect_answer", "source",
                        "adv": {..., "mcq"}}  -- note: NO "answer" field.

Everything downstream (pool building, debate seeding, grading) should never
have to know which raw shape it's dealing with. This module is the single
place that translates each raw shape into one common `FarmRecord` schema.
Verified against real data pulled directly from the official repo
(pillowsofwind/llms-believe-the-earth-is-flat), not just the README.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class FarmSchemaError(ValueError):
    """Raised when a raw record is missing fields required for its declared subset."""


VALID_SUBSETS = {"boolq", "nq1", "nq2", "truthfulqa"}


@dataclass(frozen=True)
class FarmRecord:
    """Common, subset-agnostic representation of one Farm question."""

    question_id: str
    subset: str                      # one of VALID_SUBSETS
    question: str
    gold_answer: str                 # primary gold answer, normalized to a string
    gold_answer_alternatives: list   # additional accepted phrasings (mainly TruthfulQA)
    logical_appeals: list            # list[str], the seeded incorrect rationale passages
    source: str
    raw: dict = field(repr=False)    # original record, kept for fields not yet modeled

    def to_dict(self) -> dict:
        return {
            "question_id": self.question_id,
            "subset": self.subset,
            "question": self.question,
            "gold_answer": self.gold_answer,
            "gold_answer_alternatives": self.gold_answer_alternatives,
            "logical_appeals": self.logical_appeals,
            "source": self.source,
        }


def _require(record: dict, keys: list, subset: str) -> None:
    missing = [k for k in keys if k not in record]
    if missing:
        raise FarmSchemaError(
            f"Record declared as subset='{subset}' is missing required "
            f"field(s) {missing}. Present keys: {list(record.keys())}"
        )


def _extract_logical_appeals(record: dict, subset: str) -> list:
    adv = record.get("adv", {})
    appeals = adv.get("logical", [])
    if not isinstance(appeals, list) or len(appeals) == 0:
        raise FarmSchemaError(
            f"subset='{subset}' record has no usable 'adv.logical' appeals: {appeals!r}"
        )
    return list(appeals)


def normalize_boolq(record: dict, idx: int) -> FarmRecord:
    _require(record, ["question", "answer", "source", "adv"], "boolq")
    # BoolQ questions are phrased as yes/no; the raw "answer" is a Python bool.
    # We normalize to the lowercase "yes"/"no" string a model would actually
    # produce, so grading later can treat every subset uniformly as string match.
    gold = "yes" if bool(record["answer"]) else "no"
    return FarmRecord(
        question_id=f"boolq_{idx:04d}",
        subset="boolq",
        question=record["question"],
        gold_answer=gold,
        gold_answer_alternatives=[],
        logical_appeals=_extract_logical_appeals(record, "boolq"),
        source=record["source"],
        raw=record,
    )


def _normalize_nq(record: dict, idx: int, subset: str) -> FarmRecord:
    _require(record, ["question", "answer", "source", "adv"], subset)
    return FarmRecord(
        question_id=f"{subset}_{idx:04d}",
        subset=subset,
        question=record["question"],
        gold_answer=str(record["answer"]),
        gold_answer_alternatives=[],
        logical_appeals=_extract_logical_appeals(record, subset),
        source=record["source"],
        raw=record,
    )


def normalize_nq1(record: dict, idx: int) -> FarmRecord:
    return _normalize_nq(record, idx, "nq1")


def normalize_nq2(record: dict, idx: int) -> FarmRecord:
    return _normalize_nq(record, idx, "nq2")


def normalize_truthfulqa(record: dict, idx: int) -> FarmRecord:
    _require(
        record,
        ["question", "best_answer", "correct_answer", "source", "adv"],
        "truthfulqa",
    )
    # TruthfulQA gold answers are legitimately multi-form (many phrasings are
    # all "correct"). best_answer is the primary reference; the ';'-separated
    # correct_answer field gives additional accepted phrasings for lenient
    # grading later (see src/metrics — grading is NOT implemented in this
    # module, this just preserves the alternatives so grading can use them).
    alternatives = [
        s.strip() for s in record["correct_answer"].split(";") if s.strip()
    ]
    return FarmRecord(
        question_id=f"truthfulqa_{idx:04d}",
        subset="truthfulqa",
        question=record["question"],
        gold_answer=record["best_answer"],
        gold_answer_alternatives=alternatives,
        logical_appeals=_extract_logical_appeals(record, "truthfulqa"),
        source=record["source"],
        raw=record,
    )


_NORMALIZERS = {
    "boolq": normalize_boolq,
    "nq1": normalize_nq1,
    "nq2": normalize_nq2,
    "truthfulqa": normalize_truthfulqa,
}


def normalize_record(record: dict, subset: str, idx: int) -> FarmRecord:
    """Dispatch to the correct subset-specific normalizer."""
    if subset not in VALID_SUBSETS:
        raise FarmSchemaError(
            f"Unknown subset '{subset}'. Must be one of {sorted(VALID_SUBSETS)}"
        )
    return _NORMALIZERS[subset](record, idx)
