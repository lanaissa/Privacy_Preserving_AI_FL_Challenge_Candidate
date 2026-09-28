"""Cross-silo federated learning: hospital clients that keep their patient rows private."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.features import FeatureTotals, Standardizer, feature_matrix
from src.model import TrainingConfig, init_params, loss, predict_proba, train_epochs
from src.secure_aggregation import KeyAgreement, PairwiseMasker, aggregate


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
        # Secure aggregation state: this hospital's key pair and masker, also private.
        self._key: KeyAgreement | None = None
        self._masker: PairwiseMasker | None = None

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

    # --- Messages to the server (masked when secure aggregation is on) -------------

    def start_key_agreement(self) -> int:
        """Create a fresh key pair; only the public key leaves the hospital (the server relays it)."""
        self._key = KeyAgreement()
        self._masker = None
        return self._key.public_key

    def finish_key_agreement(self, public_keys: dict[str, int]) -> None:
        """Derive one secret seed per other hospital from the relayed public keys."""
        if self._key is None:
            raise RuntimeError(f"{self.name}: key agreement was not started")
        seeds = {
            peer: self._key.shared_seed(self.name, peer, key) for peer, key in public_keys.items() if peer != self.name
        }
        self._masker = PairwiseMasker(self.name, seeds)

    def _send(self, values: np.ndarray, label: str, masked: bool) -> np.ndarray:
        if not masked:
            return np.asarray(values, dtype=float)
        if self._masker is None:
            raise RuntimeError(f"{self.name}: key agreement must be finished before sending masked messages")
        return self._masker.mask(values, label)

    def totals_message(self, label: str, masked: bool) -> np.ndarray:
        """Case count and feature totals in one vector: [n, counts..., sums..., sums of squares...]."""
        totals = self.feature_totals()
        return self._send(np.r_[self.n_cases, totals.count, totals.total, totals.total_sq], label, masked)

    def update_message(
        self,
        global_params: np.ndarray,
        epochs: int,
        config: TrainingConfig,
        rng: np.random.Generator,
        weight_by_cases: bool,
        label: str,
        masked: bool,
    ) -> np.ndarray:
        """New local weights, multiplied by the case count so the server's sum gives a weighted average."""
        params = self.local_update(global_params, epochs, config, rng)
        return self._send(params * (self.n_cases if weight_by_cases else 1), label, masked)

    def loss_message(self, params: np.ndarray, config: TrainingConfig, label: str, masked: bool) -> np.ndarray:
        """Case count times local loss, so the server's sum gives the loss over all patients."""
        return self._send(np.array([self.n_cases * self.local_loss(params, config)]), label, masked)

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
    secure: bool = True,
) -> tuple[TrainedModel, list[dict[str, Any]]]:
    """Federated Averaging. Returns the global model and one history entry per round.

    The server only ever uses the SUM of the hospitals' messages: once the case counts
    and feature totals (for scaling and weighting), then every round the case-weighted
    model weights and losses. With secure=True each message is masked, so the server
    learns only these sums, never a single hospital's numbers or size. Patient rows
    never leave a client in either mode.
    """
    if weighting not in ("cases", "equal"):
        raise ValueError(f"unknown weighting: {weighting}")
    names = sorted(clients)

    if secure:
        # Key agreement: the server collects the public keys and relays them to every hospital.
        public_keys = {name: clients[name].start_key_agreement() for name in names}
        for name in names:
            clients[name].finish_key_agreement(public_keys)

    def total(messages: list[np.ndarray]) -> np.ndarray:
        return aggregate(messages) if secure else np.sum(messages, axis=0)

    # Scaling round: total cases and summed feature totals -> one shared scaling for everyone.
    summed = total([clients[name].totals_message("feature-totals", secure) for name in names])
    n_features = (len(summed) - 1) // 3
    total_cases = summed[0]
    counts, sums, sums_sq = np.split(summed[1:], 3)
    standardizer = Standardizer(FeatureTotals(counts, sums, sums_sq))
    for name in names:
        clients[name].apply_scaling(standardizer)

    # "cases": each hospital's weights count in proportion to its patients; "equal": one vote each.
    weight_by_cases = weighting == "cases"
    divisor = total_cases if weight_by_cases else len(names)

    params = init_params(n_features)
    history = []
    for round_number in range(1, rounds + 1):
        messages = [
            clients[name].update_message(
                params,
                local_epochs,
                config,
                np.random.default_rng([seed, round_number, index]),  # reproducible shuffling per hospital and round
                weight_by_cases,
                f"round-{round_number}-weights",
                secure,
            )
            for index, name in enumerate(names)
        ]
        new_params = total(messages) / divisor
        summed_loss = total([clients[name].loss_message(new_params, config, f"round-{round_number}-loss", secure) for name in names])
        history.append(
            {
                "round": round_number,
                "training_loss": float(summed_loss[0] / total_cases),
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
