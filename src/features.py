"""Feature preparation for the readmission model.

Each case becomes a short list of numbers. Scaling uses only summary totals
(count, sum, sum of squares per feature), so in the federated setting each
hospital can share these totals instead of patient rows.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.extraction import extract_clinical_data

# Chosen with repeated cross-validation on the training set only. LVEF and creatinine
# were also tested: they scored the same (AUC 0.877 vs 0.881) and their weights took
# the opposite sign to their clinical meaning, because heart failure and CKD already
# carry that information, so the simpler set was kept.
FEATURES = [
    "age_years",
    "prior_admissions_12m",
    "heart_failure",
    "chronic_kidney_disease",
    "atrial_fibrillation",
]
_STRUCTURED = {"age_years", "prior_admissions_12m"}
_DIAGNOSES = {"heart_failure", "chronic_kidney_disease", "atrial_fibrillation"}


def raw_features(record: dict[str, Any]) -> list[float | None]:
    """Feature values for one case, in FEATURES order; None where the note has no value.

    Diagnoses and labs come from our own extraction of the note, for training cases
    too, so training and prediction features are produced the same way.
    """
    structured = record["structured_features"]
    extracted = extract_clinical_data(record["note_text"])
    values: list[float | None] = []
    for name in FEATURES:
        if name in _STRUCTURED:
            values.append(float(structured[name]))
        elif name in _DIAGNOSES:
            values.append(1.0 if name in extracted["diagnoses"] else 0.0)
        else:
            value = extracted[name]
            values.append(None if value is None else float(value))
    return values


def feature_matrix(records: list[dict[str, Any]]) -> np.ndarray:
    """Raw features for many cases as a float array; missing values are NaN."""
    rows = [[np.nan if v is None else v for v in raw_features(r)] for r in records]
    return np.array(rows, dtype=float).reshape(len(records), len(FEATURES))


@dataclass
class FeatureTotals:
    """Per-feature totals over the non-missing values: all a hospital needs to share for scaling."""

    count: np.ndarray
    total: np.ndarray
    total_sq: np.ndarray

    @classmethod
    def from_matrix(cls, x: np.ndarray) -> FeatureTotals:
        present = ~np.isnan(x)
        values = np.where(present, x, 0.0)
        return cls(present.sum(axis=0).astype(float), values.sum(axis=0), (values**2).sum(axis=0))

    def __add__(self, other: FeatureTotals) -> FeatureTotals:
        return FeatureTotals(self.count + other.count, self.total + other.total, self.total_sq + other.total_sq)


class Standardizer:
    """Fills missing values with the mean and rescales each feature to mean 0, spread 1."""

    def __init__(self, totals: FeatureTotals) -> None:
        count = np.maximum(totals.count, 1.0)
        self.mean = totals.total / count
        variance = np.maximum(totals.total_sq / count - self.mean**2, 0.0)
        std = np.sqrt(variance)
        # A feature with no spread (e.g. no patient with CKD at one hospital) is left unscaled.
        self.std = np.where(std > 0, std, 1.0)

    def transform(self, x: np.ndarray) -> np.ndarray:
        filled = np.where(np.isnan(x), self.mean, x)
        return (filled - self.mean) / self.std
