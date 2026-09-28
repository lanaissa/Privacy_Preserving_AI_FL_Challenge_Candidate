"""Evidence for the privacy summary: what the server can read with and without secure
aggregation, and what secure aggregation costs. Evaluation code only: it records the
messages the hospitals send, which is exactly the server's view.
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np

from src.features import FEATURES
from src.federated import HospitalClient, train_fedavg
from src.model import TrainingConfig
from src.secure_aggregation import FRACTION_BITS, MAX_ABS_VALUE, KeyAgreement, PairwiseMasker, aggregate, decode

_TOTALS_CKD = 1 + len(FEATURES) + FEATURES.index("chronic_kidney_disease")  # [n, counts, sums, sums_sq]
_TOTALS_AGE_COUNT = 1 + FEATURES.index("age_years")
_TOTALS_AGE_SUM = 1 + len(FEATURES) + FEATURES.index("age_years")


class RecordingClient(HospitalClient):
    """A real hospital client that keeps a copy of every message it sends (the server's view)."""

    def __init__(self, name: str, records: list[dict[str, Any]]) -> None:
        super().__init__(name, records)
        self.sent: dict[str, np.ndarray] = {}

    def totals_message(self, label, masked):
        self.sent[label] = super().totals_message(label, masked)
        return self.sent[label]

    def update_message(self, global_params, epochs, config, rng, weight_by_cases, label, masked):
        self.sent[label] = super().update_message(global_params, epochs, config, rng, weight_by_cases, label, masked)
        return self.sent[label]

    def loss_message(self, params, config, label, masked):
        self.sent[label] = super().loss_message(params, config, label, masked)
        return self.sent[label]


def _recording_clients(train: list[dict[str, Any]]) -> dict[str, RecordingClient]:
    by_site: dict[str, list[dict[str, Any]]] = {}
    for record in train:
        by_site.setdefault(record["hospital_id"], []).append(record)
    return {name: RecordingClient(name, rows) for name, rows in sorted(by_site.items())}


def _timed_training(clients, rounds, local_epochs, config, seed, secure):
    start = time.perf_counter()
    model, _ = train_fedavg(clients, rounds, local_epochs, config, seed, secure=secure)
    return model, time.perf_counter() - start


def _key_agreement_seconds(names: list[str]) -> float:
    start = time.perf_counter()
    keys = {name: KeyAgreement() for name in names}
    for name in names:
        for peer in names:
            if peer != name:
                keys[name].shared_seed(name, peer, keys[peer].public_key)
    return time.perf_counter() - start


def _masking_microseconds(n_values: int, repeats: int = 2000) -> float:
    masker = PairwiseMasker("A", {"B": b"\x01" * 32, "C": b"\x02" * 32})
    values = np.zeros(n_values)
    start = time.perf_counter()
    for i in range(repeats):
        masker.mask(values, f"timing-{i}")
    return (time.perf_counter() - start) / repeats * 1e6


def evaluate_secure_aggregation(
    train: list[dict[str, Any]], rounds: int, local_epochs: int, config: TrainingConfig, seed: int
) -> dict[str, Any]:
    """Train FedAvg with and without secure aggregation and compare utility, cost and the server's view."""
    plain_clients = _recording_clients(train)
    secure_clients = _recording_clients(train)
    names = sorted(plain_clients)
    plain_model, plain_seconds = _timed_training(plain_clients, rounds, local_epochs, config, seed, secure=False)
    secure_model, secure_seconds = _timed_training(secure_clients, rounds, local_epochs, config, seed, secure=True)

    # Utility: the two models should be the same up to fixed-point rounding.
    probability_gap = np.abs(plain_model.predict_proba(train) - secure_model.predict_proba(train)).max()

    # The server's view without secure aggregation: every hospital's numbers in the clear.
    true_rates = {
        name: float(np.mean([r["labels"]["readmission_30d"] for r in train if r["hospital_id"] == name]))
        for name in names
    }
    plain_view = {}
    for name in names:
        totals, first_update = plain_clients[name].sent["feature-totals"], plain_clients[name].sent["round-1-weights"]
        plain_view[name] = {
            "training_cases": int(totals[0]),
            "patients_with_ckd": int(totals[_TOTALS_CKD]),
            "mean_age": round(float(totals[_TOTALS_AGE_SUM] / totals[_TOTALS_AGE_COUNT]), 1),
            # All hospitals start from zero weights, so the round-1 bias moves with the hospital's readmission rate.
            "round_1_bias": round(float(first_update[0] / totals[0]), 3),
            "actual_readmission_rate": round(true_rates[name], 3),
        }
    bias_order = sorted(names, key=lambda n: plain_view[n]["round_1_bias"])
    rate_order = sorted(names, key=lambda n: true_rates[n])

    # The server's view with secure aggregation: masked messages, readable only as a sum.
    masked_totals = [secure_clients[name].sent["feature-totals"] for name in names]
    summed = aggregate(masked_totals)
    all_masked = np.concatenate([m for c in secure_clients.values() for m in c.sent.values()])
    secure_view = {
        "single_message_decoded_as_if_plain": {
            name: {
                "training_cases": float(decode(secure_clients[name].sent["feature-totals"])[0]),
                "patients_with_ckd": float(decode(secure_clients[name].sent["feature-totals"])[_TOTALS_CKD]),
            }
            for name in names
        },
        "what_the_sum_reveals": {
            "training_cases": int(round(summed[0])),
            "patients_with_ckd": int(round(summed[_TOTALS_CKD])),
            "mean_age": round(float(summed[_TOTALS_AGE_SUM] / summed[_TOTALS_AGE_COUNT]), 1),
        },
        "share_of_one_bits_in_masked_messages": round(float(np.unpackbits(all_masked.view(np.uint8)).mean()), 4),
        "masked_numbers_checked": int(all_masked.size),
    }

    n_weights = len(FEATURES) + 1
    numbers_sent = (1 + 3 * len(FEATURES)) + rounds * (n_weights + 1)
    return {
        "utility": {
            "largest_weight_difference_plain_vs_secure": float(np.abs(plain_model.params - secure_model.params).max()),
            "largest_probability_difference_on_training_cases": float(probability_gap),
            "note": "differences come only from fixed-point rounding (2^-32); metrics are identical",
        },
        "runtime": {
            "fedavg_training_seconds_plain": round(plain_seconds, 3),
            "fedavg_training_seconds_secure": round(secure_seconds, 3),
            "key_agreement_seconds_all_hospitals": round(_key_agreement_seconds(names), 3),
            "masking_one_weight_message_microseconds": round(_masking_microseconds(n_weights), 1),
        },
        "communication_per_hospital": {
            "numbers_sent_per_training": numbers_sent,
            "bytes_sent_plain": numbers_sent * 8,
            "bytes_sent_secure": numbers_sent * 8 + 256,
            "note": "masked numbers are 64-bit like plain floats, so messages do not grow; secure aggregation "
            "adds one 2048-bit public key sent (256 bytes) and two received once per training",
        },
        "server_view_without_secure_aggregation": {
            "per_hospital": plain_view,
            "inference_from_round_1_update": {
                "hospitals_ordered_by_round_1_bias": bias_order,
                "hospitals_ordered_by_actual_readmission_rate": rate_order,
                "orders_match": bias_order == rate_order,
            },
        },
        "server_view_with_secure_aggregation": secure_view,
    }


def build_privacy_summary(evaluation: dict[str, Any], rounds: int) -> dict[str, Any]:
    """privacy_summary.json: the threat model, configuration, claim and evidence for secure aggregation."""
    n_weights = len(FEATURES) + 1
    return {
        "mechanism": "secure aggregation (pairwise additive masking, Bonawitz-style without dropout recovery)",
        "implementation_status": "implemented and used for every federated training run (prototype, simulated in one process)",
        "relation_to_federated_learning": (
            "Federated learning alone keeps patient rows at the hospitals, but the server still sees each "
            "hospital's feature totals, case count, model update and loss, which leak patient-level and "
            "hospital-level information (see empirical_evidence). That is not a formal privacy guarantee. "
            "Secure aggregation adds a cryptographic guarantee about what the server sees; it is still not a "
            "differential-privacy guarantee about individual patients."
        ),
        "protected_asset": (
            "Each hospital's individual contribution: its case count, feature totals (count, sum, sum of squares "
            f"per feature), its model update in each of the {rounds} rounds and its training loss. These are "
            "computed from the hospital's patient records."
        ),
        "adversary": [
            {"who": "honest-but-curious aggregation server", "protected": True},
            {"who": "network eavesdropper", "protected": True},
            {"who": "one curious hospital", "protected": True},
            {"who": "server colluding with one hospital", "protected": True,
             "note": "learns at most the sum of the other two hospitals, not either one alone"},
            {"who": "server colluding with two hospitals", "protected": False,
             "note": "subtracting their own inputs from the total reveals the third hospital's; inherent to any "
                     "sum over three parties"},
            {"who": "actively malicious server or hospital (key substitution, dropping messages, poisoned updates)",
             "protected": False},
            {"who": "anyone receiving the global or final model", "protected": False,
             "note": "secure aggregation hides the inputs, not what the output reveals"},
        ],
        "trust_assumptions": [
            "The server and the hospitals follow the protocol (honest-but-curious).",
            "The server colludes with at most one hospital.",
            "Public keys arrive unchanged (authenticated channels, e.g. TLS with certificates, in a real "
            "deployment); the prototype assumes this.",
            "All three hospitals take part in every round; there is no dropout recovery.",
            "The Diffie-Hellman problem in the RFC 3526 group is hard and SHAKE-256 behaves like a random function.",
        ],
        "parameters": {
            "key_agreement": "Diffie-Hellman, RFC 3526 group 14 (2048-bit safe prime), generator 2; about 112-bit "
            "security (NIST SP 800-57)",
            "private_keys": "uniform in [2, q-1] from the operating system's secure random generator (Python "
            "secrets), new for every training run",
            "public_key_validation": "2 <= key <= p-2 and key^q = 1 mod p (subgroup check)",
            "seed_derivation": "SHA-256(context || shared secret || sorted pair names), one 32-byte seed per pair",
            "mask_generation": "SHAKE-256(context || seed || message label) as 64-bit words; a new label for every "
            "message, and reusing a label is refused",
            "arithmetic": f"modulo 2^64 with fixed-point encoding, {FRACTION_BITS} fraction bits (rounding at most "
            f"2^-{FRACTION_BITS + 1} per value), each value below {MAX_ABS_VALUE} in absolute value",
            "masked_messages_per_hospital": f"one totals vector ({1 + 3 * len(FEATURES)} numbers) once, then per "
            f"round the case-weighted weights ({n_weights} numbers) and the case-weighted loss (1 number)",
        },
        "privacy_claim": (
            "Against an honest-but-curious server, including one that colludes with a single hospital, each "
            "hospital's messages are indistinguishable from random numbers. The server learns only the totals over "
            "all three hospitals: the combined feature statistics, the total number of cases, the averaged model and "
            "the average loss in each round. It learns nothing about any single hospital's statistics, updates or "
            "size beyond what follows from those totals. The trained model is the same as without secure "
            "aggregation up to fixed-point rounding (about 1e-12 in the weights here)."
        ),
        "not_guaranteed": [
            "No protection of what the totals and the models reveal: with three hospitals and 120 patients the "
            "totals are still detailed, and the final model can be attacked (e.g. membership inference).",
            "No differential-privacy guarantee for individual patients.",
            "No protection against collusion of the server with two hospitals, or against active attacks.",
            "No protection of the evaluation outputs outside the protocol: experiment_summary.json reports "
            "per-hospital counts and metrics computed by the evaluation code, which has access to all data; in a "
            "real deployment each hospital would compute its own.",
        ],
        "utility_analysis": evaluation["utility"],
        "runtime_analysis": evaluation["runtime"],
        "communication_analysis": evaluation["communication_per_hospital"],
        "empirical_evidence": {
            "without_secure_aggregation": evaluation["server_view_without_secure_aggregation"],
            "with_secure_aggregation": evaluation["server_view_with_secure_aggregation"],
        },
        "remaining_attack_surface": [
            "The aggregated totals and the global model in every round (visible to the server and all hospitals).",
            "The final model and its predictions.",
            "Collusion of the server with two hospitals.",
            "Key substitution by an active server, since public keys are not authenticated in the prototype.",
            "Timing and other side channels, not considered in a single-process simulation.",
        ],
        "limitations": [
            "Prototype simulated in one process: no real network, TLS or key authentication.",
            "No dropout recovery: if a hospital is missing, its masks do not cancel and the round fails. The full "
            "protocol of Bonawitz et al. (2017) recovers using secret sharing.",
            "Only three hospitals, so the sum itself is detailed and collusion of two reveals the third.",
            "A natural next step is to combine secure aggregation with differential privacy (noise added to the "
            "updates before masking), so that the revealed sums also protect individual patients.",
        ],
    }
