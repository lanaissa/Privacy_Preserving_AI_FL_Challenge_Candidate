#!/usr/bin/env python3
"""Standard challenge entry point: de-identification, extraction and federated readmission prediction."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.deid import detect_pii, render_deidentified
from src.experiment import run_experiment
from src.extraction import extract_clinical_data


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_labels(path: Path, records: list[dict[str, Any]]) -> dict[str, int]:
    """Readmission labels for the input cases; every input case must have one."""
    labels = {row["case_id"]: int(row["readmission_30d"]) for row in read_jsonl(path)}
    missing = [record["case_id"] for record in records if record["case_id"] not in labels]
    if missing:
        raise ValueError(f"--ground-truth has no label for {len(missing)} input cases, e.g. {missing[0]}")
    return labels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifacts-dir", type=Path, required=True)
    parser.add_argument(
        "--ground-truth",
        type=Path,
        help="optional labels for the input cases; only used to add validation metrics to experiment_summary.json",
    )
    args = parser.parse_args()

    train_records = read_jsonl(args.train)
    evaluation_records = read_jsonl(args.input)
    labels = read_labels(args.ground_truth, evaluation_records) if args.ground_truth else None
    model, experiment_summary = run_experiment(train_records, evaluation_records if labels else None, labels)
    probabilities = model.predict_proba(evaluation_records)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record, probability in zip(evaluation_records, probabilities):
            spans = detect_pii(record["note_text"])
            prediction = {
                "case_id": record["case_id"],
                "pii_entities": spans,
                "deidentified_text": render_deidentified(record["note_text"], spans),
                "extracted_clinical_data": extract_clinical_data(record["note_text"]),
                "readmission_probability": float(np.clip(probability, 0.0, 1.0)),
            }
            handle.write(json.dumps(prediction, ensure_ascii=False) + "\n")

    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    (args.artifacts_dir / "experiment_summary.json").write_text(
        json.dumps(experiment_summary, indent=2), encoding="utf-8"
    )
    privacy_summary = {
        "mechanism": None,
        "threat_model": "TODO",
        "privacy_guarantee": "TODO",
        "utility_analysis": "TODO",
        "limitations": "TODO",
    }
    (args.artifacts_dir / "privacy_summary.json").write_text(
        json.dumps(privacy_summary, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
