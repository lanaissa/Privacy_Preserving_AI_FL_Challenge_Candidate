"""Readmission experiment: local vs federated vs centralized, evaluated the same way.

Every setting uses the same model, features, scaling, training settings and amount
of training (100 passes over the data), so the only difference is who sees which rows.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src.features import FEATURES
from src.federated import TrainedModel, partition_by_hospital, train_centralized, train_fedavg
from src.model import TrainingConfig

SEEDS = [7, 19, 43]
CONFIG = TrainingConfig()
ROUNDS = 50
LOCAL_EPOCHS = 2
EPOCHS = ROUNDS * LOCAL_EPOCHS  # local and centralized models get the same number of passes
CV_FOLDS = 5
SETTINGS = ("local", "federated", "centralized")


def _metrics(y: np.ndarray, p: np.ndarray) -> dict[str, Any]:
    """The evaluator's metrics. AUC and average precision need both outcomes present."""
    both = len(set(y.tolist())) == 2
    return {
        "n": int(len(y)),
        "readmitted": int(y.sum()),
        "roc_auc": round(float(roc_auc_score(y, p)), 3) if both else None,
        "average_precision": round(float(average_precision_score(y, p)), 3) if both else None,
        "brier_score": round(float(brier_score_loss(y, p)), 3),
    }


def _overall_and_by_site(y: np.ndarray, p: np.ndarray, sites: np.ndarray) -> dict[str, Any]:
    result = _metrics(y, p)
    result["by_site"] = {site: _metrics(y[sites == site], p[sites == site]) for site in sorted(set(sites))}
    return result


def _mean_over_seeds(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Average each metric over the seeds and add its spread (standard deviation)."""
    summary: dict[str, Any] = {}
    for key in ("roc_auc", "average_precision", "brier_score"):
        values = [run[key] for run in runs if run[key] is not None]
        summary[key] = round(float(np.mean(values)), 3) if values else None
        summary[f"{key}_std_over_seeds"] = round(float(np.std(values)), 3) if values else None
    summary["n"], summary["readmitted"] = runs[0]["n"], runs[0]["readmitted"]
    if "by_site" in runs[0]:
        summary["by_site"] = {
            site: _mean_over_seeds([run["by_site"][site] for run in runs]) for site in runs[0]["by_site"]
        }
    return summary


def _train_all(records: list[dict[str, Any]], seed: int) -> dict[str, Any]:
    """Train the three settings on the same records with the same seed."""
    local = {
        name: client.train_local_model(EPOCHS, CONFIG, seed)
        for name, client in partition_by_hospital(records).items()
    }
    federated, history = train_fedavg(partition_by_hospital(records), ROUNDS, LOCAL_EPOCHS, CONFIG, seed)
    centralized = train_centralized(records, EPOCHS, CONFIG, seed)
    return {"local": local, "federated": federated, "centralized": centralized, "history": history}


def _predict(models: dict[str, Any], setting: str, records: list[dict[str, Any]]) -> np.ndarray:
    """Local: each patient is scored by their own hospital's model."""
    if setting != "local":
        return models[setting].predict_proba(records)
    return np.array([models["local"][r["hospital_id"]].predict_proba([r])[0] for r in records])


def cross_validate(train: list[dict[str, Any]]) -> dict[str, Any]:
    """Stratified 5-fold cross-validation on the training set, repeated for each seed.

    Folds are balanced by hospital and outcome together, so every fold contains
    every hospital. Metrics use the pooled out-of-fold predictions of each seed.
    """
    y = np.array([r["labels"]["readmission_30d"] for r in train], dtype=float)
    sites = np.array([r["hospital_id"] for r in train])
    strata = [f"{site}_{int(label)}" for site, label in zip(sites, y)]
    runs: dict[str, list[dict[str, Any]]] = {setting: [] for setting in SETTINGS}
    for seed in SEEDS:
        predictions = {setting: np.zeros(len(train)) for setting in SETTINGS}
        for train_idx, test_idx in StratifiedKFold(CV_FOLDS, shuffle=True, random_state=seed).split(train, strata):
            models = _train_all([train[i] for i in train_idx], seed)
            test_records = [train[i] for i in test_idx]
            for setting in SETTINGS:
                predictions[setting][test_idx] = _predict(models, setting, test_records)
        for setting in SETTINGS:
            runs[setting].append(_overall_and_by_site(y, predictions[setting], sites))
    return {setting: _mean_over_seeds(runs[setting]) for setting in SETTINGS}


def validate(train: list[dict[str, Any]], validation: list[dict[str, Any]], labels: dict[str, int]) -> dict[str, Any]:
    """Train on the full training set and score the labelled validation cases, for each seed."""
    y = np.array([labels[r["case_id"]] for r in validation], dtype=float)
    sites = np.array([r["hospital_id"] for r in validation])
    runs: dict[str, list[dict[str, Any]]] = {setting: [] for setting in SETTINGS}
    for seed in SEEDS:
        models = _train_all(train, seed)
        for setting in SETTINGS:
            runs[setting].append(_overall_and_by_site(y, _predict(models, setting, validation), sites))
    return {setting: _mean_over_seeds(runs[setting]) for setting in SETTINGS}


def _site_profile(train: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-hospital counts that show how the sites differ (non-IID)."""
    profile = {}
    for name, client in partition_by_hospital(train).items():
        totals = client.feature_totals()
        rows = [r for r in train if r["hospital_id"] == name]
        profile[name] = {
            "training_cases": client.n_cases,
            "readmission_rate": round(float(np.mean([r["labels"]["readmission_30d"] for r in rows])), 3),
            "mean_age": round(float(totals.total[0] / totals.count[0]), 1),
            "prior_admissions_total": int(totals.total[1]),
            "patients_with_heart_failure": int(totals.total[2]),
            "patients_with_ckd": int(totals.total[3]),
            "patients_with_af": int(totals.total[4]),
        }
    return profile


def run_experiment(
    train: list[dict[str, Any]],
    validation: list[dict[str, Any]] | None = None,
    labels: dict[str, int] | None = None,
) -> tuple[TrainedModel, dict[str, Any]]:
    """Return the final federated model and the experiment summary.

    Validation metrics are only added when validation labels are given; the hidden
    test run has no labels, so it reports the cross-validation results only.
    """
    final_model, history = train_fedavg(partition_by_hospital(train), ROUNDS, LOCAL_EPOCHS, CONFIG, SEEDS[0])
    cv = cross_validate(train)
    val = validate(train, validation, labels) if validation is not None and labels is not None else None
    n_weights = len(FEATURES) + 1

    def results(setting: str) -> dict[str, Any]:
        return {"cross_validation": cv[setting], "validation": val[setting] if val else None}

    summary = {
        "implementation": "fedavg",
        "final_predictions_from": f"federated model trained on all training cases with seed {SEEDS[0]}",
        "random_seeds": SEEDS,
        "features": FEATURES,
        "feature_sources": {
            "structured_features": ["age_years", "prior_admissions_12m"],
            "extracted_from_note": ["heart_failure", "chronic_kidney_disease", "atrial_fibrillation"],
        },
        "model": {
            "algorithm": "logistic_regression (numpy, same code for all settings)",
            "optimizer": "mini-batch gradient descent",
            "learning_rate": CONFIG.learning_rate,
            "l2_penalty": CONFIG.l2,
            "batch_size": CONFIG.batch_size,
            "scaling": "mean 0 / spread 1, from summed per-hospital feature totals",
        },
        "evaluation": {
            "cross_validation": f"stratified {CV_FOLDS}-fold on the training set (balanced by hospital and outcome), "
            f"one run per seed; metrics are the mean over seeds with their spread",
            "validation": "trained on all training cases, scored on the labelled validation cases"
            if val
            else "not available (no labels given, e.g. the hidden test run)",
        },
        "local_models": {
            "algorithm": "logistic_regression",
            "epochs": EPOCHS,
            "scaling": "each hospital's own totals (no collaboration at all)",
            "note": "each patient is scored by their own hospital's model",
            **results("local"),
        },
        "federated_model": {
            "algorithm": "FedAvg",
            "rounds": ROUNDS,
            "local_epochs": LOCAL_EPOCHS,
            "client_weighting": "number_of_training_cases",
            **results("federated"),
            "convergence": {
                "training_loss_by_round": [round(h["training_loss"], 4) for h in history],
                "weight_change_by_round": [round(h["update_size"], 4) for h in history],
            },
            "final_weights": dict(zip(["bias", *FEATURES], [round(float(v), 3) for v in final_model.params])),
        },
        "centralized_model": {
            "algorithm": "logistic_regression",
            "epochs": EPOCHS,
            "note": "reference only: pools all hospitals' rows, which the real setting forbids",
            **results("centralized"),
        },
        "communication": {
            "raw_rows_transferred": False,
            "once_per_training_client_to_server": f"number of training cases (1 number) and feature totals "
            f"(count, sum, sum of squares for {len(FEATURES)} features = {3 * len(FEATURES)} numbers)",
            "once_per_training_server_to_client": f"shared scaling (mean and spread, {2 * len(FEATURES)} numbers)",
            "each_round_server_to_client": f"global model weights ({n_weights} numbers)",
            "each_round_client_to_server": f"updated model weights ({n_weights} numbers) and one training-loss value",
            "numbers_per_client_per_training": 1 + 3 * len(FEATURES) + ROUNDS * (n_weights + 1),
        },
        "non_iid_analysis": {
            "site_profile": _site_profile(train),
            "observations": [
                "Readmission rates differ by hospital (Chennai highest, Hyderabad lowest).",
                "Berlin has no training patients with CKD, so its local model cannot learn CKD at all; "
                "the federated model learns it from Chennai and Hyderabad.",
                "Almost all prior admissions are at Chennai, so FedAvg's averaging slightly dilutes that "
                "feature compared with pooling (client drift).",
                "Berlin patients are the hardest to rank for every setting.",
            ],
        },
        "limitations": [
            "Only 120 training and 30 validation cases; validation has 6 readmissions (1 at Hyderabad), "
            "so per-site validation metrics are very noisy. Cross-validation is the more reliable comparison.",
            "Features were chosen with cross-validation on the same training set, which makes the "
            "cross-validation scores somewhat optimistic.",
            "Model weights are shared in the clear; federated learning alone is not a formal privacy guarantee "
            "(see privacy_summary.json).",
            "Hospitals are nearly equal in size, so weighting by cases and equal weighting give almost the same model.",
        ],
    }
    return final_model, summary
