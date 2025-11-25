from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence


@dataclass
class EvaluationResult:
    precision: float
    recall: float
    f1: float
    coverage: float


def _safe_div(num: float, denom: float) -> float:
    return num / denom if denom else 0.0


def compute_metrics(predictions: Dict[str, Sequence[str]], labels: Dict[str, Sequence[str]]) -> EvaluationResult:
    true_positive = 0
    false_positive = 0
    false_negative = 0
    covered = 0

    for case_id, expected in labels.items():
        predicted = set(predictions.get(case_id, []))
        expected_set = set(expected)
        covered += 1 if predicted else 0
        true_positive += len(predicted & expected_set)
        false_positive += len(predicted - expected_set)
        false_negative += len(expected_set - predicted)

    precision = _safe_div(true_positive, true_positive + false_positive)
    recall = _safe_div(true_positive, true_positive + false_negative)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    coverage = _safe_div(covered, len(labels))
    return EvaluationResult(precision=precision, recall=recall, f1=f1, coverage=coverage)

