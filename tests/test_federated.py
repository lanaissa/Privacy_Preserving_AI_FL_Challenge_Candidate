import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

import run_submission
from src.experiment import run_experiment
from src.features import FEATURES, FeatureTotals, Standardizer, feature_matrix, raw_features
from src.federated import HospitalClient, partition_by_hospital, train_centralized, train_fedavg
from src.model import TrainingConfig, gradient, init_params, loss, train_epochs

ROOT = Path(__file__).resolve().parent.parent
TRAIN = [json.loads(line) for line in open(ROOT / "data" / "train.jsonl")]
N_WEIGHTS = len(FEATURES) + 1


def scaled_training_data() -> tuple[np.ndarray, np.ndarray]:
    x = feature_matrix(TRAIN)
    y = np.array([r["labels"]["readmission_30d"] for r in TRAIN], dtype=float)
    return Standardizer(FeatureTotals.from_matrix(x)).transform(x), y


# --- Features --------------------------------------------------------------------

# Diagnoses come from our own extraction of the note, not from the training labels,
# so training and prediction features are built the same way (labels are not even needed).
def test_features_come_from_the_note_not_the_labels() -> None:
    record = {
        "structured_features": {"age_years": 70, "prior_admissions_12m": 2},
        "note_text": "Dx: HFrEF; CKD-3. The patient denies AF.",
    }
    assert raw_features(record) == [70.0, 2.0, 1.0, 1.0, 0.0]


# Adding the hospitals' totals gives exactly the totals of the pooled data.
def test_hospital_totals_add_up_to_pooled_totals() -> None:
    parts = [c.feature_totals() for c in partition_by_hospital(TRAIN).values()]
    combined = parts[0] + parts[1] + parts[2]
    pooled = FeatureTotals.from_matrix(feature_matrix(TRAIN))
    for field in ("count", "total", "total_sq"):
        assert np.allclose(getattr(combined, field), getattr(pooled, field))


# Scaling gives mean 0 / spread 1, fills missing values with the mean,
# and leaves a feature with no spread unscaled instead of dividing by zero.
def test_standardizer() -> None:
    x = np.array([[1.0, 5.0], [3.0, 5.0], [np.nan, 5.0]])
    scaled = Standardizer(FeatureTotals.from_matrix(x)).transform(x)
    assert np.allclose(scaled[:2, 0], [-1.0, 1.0])
    assert scaled[2, 0] == 0.0  # missing -> mean -> 0
    assert np.all(scaled[:, 1] == 0.0) and not np.isnan(scaled).any()


# --- Model -----------------------------------------------------------------------

# The gradient formula matches a slow numerical estimate.
def test_gradient_matches_numerical_estimate() -> None:
    x, y = scaled_training_data()
    params = np.random.default_rng(0).normal(size=N_WEIGHTS) * 0.3
    eps = 1e-6
    numerical = [
        (loss(params + eps * e, x, y, 0.01) - loss(params - eps * e, x, y, 0.01)) / (2 * eps)
        for e in np.eye(N_WEIGHTS)
    ]
    assert np.allclose(numerical, gradient(params, x, y, 0.01), atol=1e-6)


# Our hand-written logistic regression gives the same weights as scikit-learn's.
def test_model_matches_scikit_learn() -> None:
    x, y = scaled_training_data()
    config = TrainingConfig(learning_rate=0.5, l2=0.01, batch_size=len(y))
    ours = train_epochs(init_params(len(FEATURES)), x, y, 5000, config, np.random.default_rng(7))
    reference = LogisticRegression(C=1 / (0.01 * len(y)), max_iter=10_000, tol=1e-10).fit(x, y)
    assert np.allclose(ours, np.r_[reference.intercept_, reference.coef_[0]], atol=1e-3)


# Training lowers the loss, returns new weights without changing the input,
# and the same seed always gives the same result.
def test_training_is_deterministic_and_lowers_loss() -> None:
    x, y = scaled_training_data()
    start = init_params(len(FEATURES))
    first = train_epochs(start, x, y, 50, TrainingConfig(), np.random.default_rng(7))
    second = train_epochs(start, x, y, 50, TrainingConfig(), np.random.default_rng(7))
    assert np.array_equal(first, second)
    assert np.all(start == 0)
    assert loss(first, x, y, 0.01) < loss(start, x, y, 0.01)


# --- Hospital clients: data stays inside ------------------------------------------

# One client per hospital, with the right number of cases.
def test_partition_by_hospital() -> None:
    clients = partition_by_hospital(TRAIN)
    assert {name: c.n_cases for name, c in clients.items()} == {
        "BERLIN_NODE": 42,
        "CHENNAI_NODE": 39,
        "HYDERABAD_NODE": 39,
    }


# A hospital refuses records from another hospital.
def test_client_rejects_other_hospitals_records() -> None:
    chennai_record = next(r for r in TRAIN if r["hospital_id"] == "CHENNAI_NODE")
    with pytest.raises(ValueError):
        HospitalClient("BERLIN_NODE", [chennai_record])


# A hospital cannot train before the server has sent the shared scaling.
def test_client_needs_scaling_before_training() -> None:
    client = partition_by_hospital(TRAIN)["BERLIN_NODE"]
    with pytest.raises(RuntimeError):
        client.local_update(init_params(len(FEATURES)), 1, TrainingConfig(), np.random.default_rng(0))


class SpyClient(HospitalClient):
    """A real hospital client that records every message it sends to the server."""

    def __init__(self, name, records):
        super().__init__(name, records)
        self.sent = []

    def start_key_agreement(self):
        value = super().start_key_agreement()
        self.sent.append(("public_key", value))
        return value

    def totals_message(self, *args, **kwargs):
        value = super().totals_message(*args, **kwargs)
        self.sent.append(("totals", value))
        return value

    def update_message(self, *args, **kwargs):
        value = super().update_message(*args, **kwargs)
        self.sent.append(("update", value))
        return value

    def loss_message(self, *args, **kwargs):
        value = super().loss_message(*args, **kwargs)
        self.sent.append(("loss", value))
        return value


def spy_clients() -> dict[str, SpyClient]:
    by_site = {}
    for record in TRAIN:
        by_site.setdefault(record["hospital_id"], []).append(record)
    return {name: SpyClient(name, rows) for name, rows in by_site.items()}


# During a full FedAvg run (without masking, so the values are readable), a hospital only
# ever sends: one totals vector (case count + 3 totals per feature), model weights and
# single loss values. No patient rows or labels leave the hospital.
def test_fedavg_only_receives_summaries_from_hospitals() -> None:
    clients = spy_clients()
    train_fedavg(clients, rounds=3, local_epochs=1, config=TrainingConfig(), seed=7, secure=False)
    expected_shape = {"totals": (1 + 3 * len(FEATURES),), "update": (N_WEIGHTS,), "loss": (1,)}
    for client in clients.values():
        kinds = [kind for kind, _ in client.sent]
        assert kinds.count("totals") == 1 and kinds.count("update") == 3 and kinds.count("loss") == 3
        for kind, value in client.sent:
            assert value.shape == expected_shape[kind]  # never a data matrix


# --- FedAvg ----------------------------------------------------------------------

# Proof that the averaging is correct: with one full-batch local step, FedAvg gives
# exactly the same weights as training on the pooled data.
def test_fedavg_equals_pooled_training_with_one_full_batch_step() -> None:
    config = TrainingConfig(learning_rate=0.5, batch_size=10_000)
    federated, _ = train_fedavg(
        partition_by_hospital(TRAIN), rounds=100, local_epochs=1, config=config, seed=7, secure=False
    )
    x, y = scaled_training_data()
    pooled = train_epochs(init_params(len(FEATURES)), x, y, 100, config, np.random.default_rng(7))
    assert np.allclose(federated.params, pooled, atol=1e-10)


class FixedClient:
    """A stand-in hospital that always returns the same weights, to check the averaging maths."""

    def __init__(self, n_cases, weights):
        self.n_cases = n_cases
        self._weights = np.array(weights, dtype=float)

    def totals_message(self, label, masked):
        return np.array([self.n_cases, 1.0, 0.0, 1.0])  # case count + totals for one feature

    def apply_scaling(self, standardizer):
        pass

    def update_message(self, global_params, epochs, config, rng, weight_by_cases, label, masked):
        return self._weights * (self.n_cases if weight_by_cases else 1)

    def loss_message(self, params, config, label, masked):
        return np.array([0.0])


# The server weights each hospital by its number of cases, or gives equal votes if asked.
@pytest.mark.parametrize("weighting, expected", [("cases", [2.5, 5.0]), ("equal", [2.0, 4.0])])
def test_fedavg_weighting(weighting: str, expected: list[float]) -> None:
    clients = {"A": FixedClient(10, [1.0, 2.0]), "B": FixedClient(30, [3.0, 6.0])}
    model, _ = train_fedavg(
        clients, rounds=1, local_epochs=1, config=TrainingConfig(), seed=0, weighting=weighting, secure=False
    )
    assert np.allclose(model.params, expected)  # cases: 10/40 * A + 30/40 * B


# FedAvg converges: the loss falls and the weights almost stop changing.
def test_fedavg_converges() -> None:
    _, history = train_fedavg(partition_by_hospital(TRAIN), 50, 2, TrainingConfig(), seed=7)
    assert history[-1]["training_loss"] < history[0]["training_loss"] - 0.15
    assert history[-1]["update_size"] < 0.05


# Same seed -> same model; different seeds -> nearly the same model (seed stability).
def test_fedavg_seed_stability() -> None:
    a, _ = train_fedavg(partition_by_hospital(TRAIN), 50, 2, TrainingConfig(), seed=7)
    b, _ = train_fedavg(partition_by_hospital(TRAIN), 50, 2, TrainingConfig(), seed=7)
    c, _ = train_fedavg(partition_by_hospital(TRAIN), 50, 2, TrainingConfig(), seed=19)
    assert np.array_equal(a.params, b.params)
    assert np.abs(a.params - c.params).max() < 0.1


# Non-IID example: Berlin has no CKD patients, so its local model cannot learn CKD,
# while the federated model learns it from the other hospitals.
def test_berlin_alone_cannot_learn_ckd() -> None:
    ckd = 1 + FEATURES.index("chronic_kidney_disease")
    local = partition_by_hospital(TRAIN)["BERLIN_NODE"].train_local_model(100, TrainingConfig(), seed=7)
    federated, _ = train_fedavg(partition_by_hospital(TRAIN), 50, 2, TrainingConfig(), seed=7)
    assert local.params[ckd] == 0.0
    assert federated.params[ckd] > 0.5


# Federated and centralized end up with nearly the same model (logistic regression has one best answer).
def test_federated_close_to_centralized() -> None:
    federated, _ = train_fedavg(partition_by_hospital(TRAIN), 50, 2, TrainingConfig(), seed=7)
    centralized = train_centralized(TRAIN, 100, TrainingConfig(), seed=7)
    assert np.abs(federated.predict_proba(TRAIN) - centralized.predict_proba(TRAIN)).max() < 0.05


# --- Experiment summary and entry point -------------------------------------------

@pytest.fixture(scope="module")
def experiment_with_labels():
    inputs = [json.loads(line) for line in open(ROOT / "data" / "validation_inputs.jsonl")]
    labels = run_submission.read_labels(ROOT / "data" / "validation_ground_truth.jsonl", inputs)
    return run_experiment(TRAIN, inputs, labels)


# The summary contains everything SUBMISSION_SCHEMA.md asks for, for all three settings.
def test_summary_has_required_fields(experiment_with_labels) -> None:
    _, summary = experiment_with_labels
    for key in ("random_seeds", "features", "model", "communication", "non_iid_analysis", "limitations"):
        assert summary[key]
    assert {"rounds", "local_epochs", "client_weighting", "convergence"} <= set(summary["federated_model"])
    assert summary["communication"]["raw_rows_transferred"] is False
    for setting in ("local_models", "federated_model", "centralized_model"):
        for part in ("cross_validation", "validation"):
            metrics = summary[setting][part]
            assert set(metrics["by_site"]) == {"BERLIN_NODE", "CHENNAI_NODE", "HYDERABAD_NODE"}
            assert 0.0 <= metrics["roc_auc"] <= 1.0


# The expected ranking from cross-validation: local is worse, federated is close to centralized.
def test_local_worse_than_federated_close_to_centralized(experiment_with_labels) -> None:
    _, summary = experiment_with_labels
    auc = {s: summary[s]["cross_validation"]["roc_auc"] for s in ("local_models", "federated_model", "centralized_model")}
    assert auc["local_models"] < auc["federated_model"]
    assert abs(auc["federated_model"] - auc["centralized_model"]) < 0.02


# --ground-truth must have a label for every input case.
def test_ground_truth_must_cover_every_input(tmp_path: Path) -> None:
    labels_file = tmp_path / "labels.jsonl"
    labels_file.write_text(json.dumps({"case_id": "A", "readmission_30d": 1}) + "\n")
    with pytest.raises(ValueError):
        run_submission.read_labels(labels_file, [{"case_id": "A"}, {"case_id": "B"}])


# The hidden-test command (no labels) works: one valid prediction per case, and the
# summary reports cross-validation only.
def test_entry_point_without_labels(tmp_path: Path) -> None:
    output = tmp_path / "predictions.jsonl"
    artifacts = tmp_path / "artifacts"
    subprocess.run(
        [sys.executable, "run_submission.py", "--train", "data/train.jsonl", "--input", "data/validation_inputs.jsonl",
         "--output", str(output), "--artifacts-dir", str(artifacts)],
        cwd=ROOT, check=True,
    )
    predictions = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(predictions) == 30
    assert all(0.0 <= p["readmission_probability"] <= 1.0 for p in predictions)
    summary = json.loads((artifacts / "experiment_summary.json").read_text())
    assert summary["federated_model"]["validation"] is None
    assert summary["federated_model"]["cross_validation"]["roc_auc"] > 0.5
