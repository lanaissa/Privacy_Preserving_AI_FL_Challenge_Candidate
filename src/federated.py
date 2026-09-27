"""Cross-silo federated learning: hospital clients that keep their patient rows private."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.features import FeatureTotals, Standardizer, feature_matrix
from src.model import TrainingConfig, init_params, loss, predict_proba, train_epochs


@dataclass
class TrainedModel:
    """Model weights plus the scaling they were trained with; both are needed to predict."""

    params: np.ndarray
    standardizer: Standardizer

    def predict_proba(self, records: list[dict[str, Any]]) -> np.ndarray:
        return predict_proba(self.params, self.standardizer.transform(feature_matrix(records)))


class HospitalClient:
    """One hospital. Its patient rows stay inside this object.

    The public methods only return summaries (the number of cases and feature
    totals); nothing returns individual rows. The server and the other hospitals
    only ever talk to a client through these methods.
    """

    def __init__(self, name: str, records: list[dict[str, Any]]) -> None:
        if any(record["hospital_id"] != name for record in records):
            raise ValueError(f"{name} received records from another hospital")
        self.name = name
        # Private: raw features and labels never leave this object.
        self._x_raw = feature_matrix(records)
        self._y = np.array([int(record["labels"]["readmission_30d"]) for record in records], dtype=float)
        self._x: np.ndarray | None = None

    @property
    def n_cases(self) -> int:
        """Number of training cases; shared with the server for client weighting."""
        return len(self._y)

    def feature_totals(self) -> FeatureTotals:
        """Count, sum and sum of squares per feature: the only data summary shared for scaling."""
        return FeatureTotals.from_matrix(self._x_raw)

    def apply_scaling(self, standardizer: Standardizer) -> None:
        """Scale the local features with the scaling the server sends back."""
        self._x = standardizer.transform(self._x_raw)

    def local_update(
        self, global_params: np.ndarray, epochs: int, config: TrainingConfig, rng: np.random.Generator
    ) -> np.ndarray:
        """Train a copy of the global model on local data and return only the new weights."""
        if self._x is None:
            raise RuntimeError(f"{self.name}: scaling must be applied before training")
        return train_epochs(global_params, self._x, self._y, epochs, config, rng)

    def local_loss(self, params: np.ndarray, config: TrainingConfig) -> float:
        """One number: the model's error on local data, reported for convergence tracking."""
        if self._x is None:
            raise RuntimeError(f"{self.name}: scaling must be applied before evaluating")
        return loss(params, self._x, self._y, config.l2)

    def train_local_model(self, epochs: int, config: TrainingConfig, seed: int) -> TrainedModel:
        """No-collaboration baseline: scaling and training use this hospital's data only."""
        standardizer = Standardizer(self.feature_totals())
        x = standardizer.transform(self._x_raw)
        params = train_epochs(init_params(x.shape[1]), x, self._y, epochs, config, np.random.default_rng(seed))
        return TrainedModel(params, standardizer)


def train_fedavg(
    clients: dict[str, HospitalClient],
    rounds: int,
    local_epochs: int,
    config: TrainingConfig,
    seed: int,
    weighting: str = "cases",
) -> tuple[TrainedModel, list[dict[str, Any]]]:
    """Federated Averaging. Returns the global model and one history entry per round.

    What crosses the hospital boundary: feature totals and case counts (once, for
    scaling and weighting), model weights (both directions, every round) and one
    loss value per round. Patient rows never leave a client.
    """
    names = sorted(clients)
    # Scaling round: the server adds up the hospitals' totals and sends one scaling back.
    totals = clients[names[0]].feature_totals()
    for name in names[1:]:
        totals = totals + clients[name].feature_totals()
    standardizer = Standardizer(totals)
    for name in names:
        clients[name].apply_scaling(standardizer)

    # "cases": a hospital's vote is proportional to its number of patients; "equal": one vote each.
    counts = np.array([clients[name].n_cases for name in names], dtype=float)
    if weighting == "cases":
        client_weights = counts / counts.sum()
    elif weighting == "equal":
        client_weights = np.full(len(names), 1.0 / len(names))
    else:
        raise ValueError(f"unknown weighting: {weighting}")

    params = init_params(len(totals.count))
    history = []
    for round_number in range(1, rounds + 1):
        updates = []
        for index, name in enumerate(names):
            # Each hospital gets its own reproducible shuffling for each round.
            rng = np.random.default_rng([seed, round_number, index])
            updates.append(clients[name].local_update(params, local_epochs, config, rng))
        new_params = np.sum([w * u for w, u in zip(client_weights, updates)], axis=0)
        local_losses = [clients[name].local_loss(new_params, config) for name in names]
        history.append(
            {
                "round": round_number,
                "training_loss": float(np.dot(counts / counts.sum(), local_losses)),
                "update_size": float(np.linalg.norm(new_params - params)),
            }
        )
        params = new_params
    return TrainedModel(params, standardizer), history


def train_centralized(records: list[dict[str, Any]], epochs: int, config: TrainingConfig, seed: int) -> TrainedModel:
    """Reference only: pools every hospital's rows in one place, which the real setting forbids.

    Same model, features, scaling and settings as FedAvg, so the only difference is pooling.
    """
    x_raw = feature_matrix(records)
    y = np.array([int(record["labels"]["readmission_30d"]) for record in records], dtype=float)
    standardizer = Standardizer(FeatureTotals.from_matrix(x_raw))
    x = standardizer.transform(x_raw)
    params = train_epochs(init_params(x.shape[1]), x, y, epochs, config, np.random.default_rng(seed))
    return TrainedModel(params, standardizer)


def partition_by_hospital(records: list[dict[str, Any]]) -> dict[str, HospitalClient]:
    """Split training records into one client per hospital, sorted by hospital name."""
    by_hospital: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_hospital.setdefault(record["hospital_id"], []).append(record)
    return {name: HospitalClient(name, rows) for name, rows in sorted(by_hospital.items())}
