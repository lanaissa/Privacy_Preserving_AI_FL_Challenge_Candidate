import json
import random
from decimal import Decimal, getcontext
from pathlib import Path

import numpy as np
import pytest

from src.experiment import CONFIG, LOCAL_EPOCHS, ROUNDS
from src.features import FEATURES
from src.federated import HospitalClient, partition_by_hospital, train_fedavg
from src.privacy_evaluation import build_privacy_summary, evaluate_secure_aggregation
from src.secure_aggregation import (
    MAX_ABS_VALUE,
    P,
    Q,
    KeyAgreement,
    PairwiseMasker,
    _mask,
    aggregate,
    decode,
    encode,
    is_valid_public_key,
)

ROOT = Path(__file__).resolve().parent.parent
with open(ROOT / "data" / "train.jsonl") as handle:
    TRAIN = [json.loads(line) for line in handle]
NAMES = ["BERLIN_NODE", "CHENNAI_NODE", "HYDERABAD_NODE"]


def make_maskers(names: list[str]) -> dict[str, PairwiseMasker]:
    keys = {name: KeyAgreement() for name in names}
    return {
        name: PairwiseMasker(name, {p: keys[name].shared_seed(name, p, keys[p].public_key) for p in names if p != name})
        for name in names
    }


# --- Key agreement ---------------------------------------------------------------

# The prime is exactly RFC 3526's 2048-bit prime, rebuilt from its official formula with the digits of pi.
def test_prime_matches_rfc_3526() -> None:
    getcontext().prec = 720

    def arctan_inverse(x: int) -> Decimal:
        x_dec, total, n, sign = Decimal(x), Decimal(1) / x, 1, -1
        term = total
        while True:
            term /= x_dec * x_dec
            n += 2
            if term / n < Decimal(10) ** -715:
                return total
            total += sign * term / n
            sign = -sign

    pi = 16 * arctan_inverse(5) - 4 * arctan_inverse(239)
    assert P == 2**2048 - 2**1984 - 1 + 2**64 * (int(Decimal(2**1918) * pi) + 124476)


# P is a safe prime: both P and Q = (P - 1) / 2 are prime (Miller-Rabin).
def test_prime_is_safe_prime() -> None:
    def probably_prime(n: int) -> bool:
        d, r = n - 1, 0
        while d % 2 == 0:
            d, r = d // 2, r + 1
        rng = random.Random(0)
        for _ in range(20):
            x = pow(rng.randrange(2, n - 2), d, n)
            if x in (1, n - 1):
                continue
            for _ in range(r - 1):
                x = pow(x, 2, n)
                if x == n - 1:
                    break
            else:
                return False
        return True

    assert probably_prime(P) and probably_prime(Q)


# Both hospitals of a pair compute the same seed; the three pairs get different seeds.
def test_pairs_agree_on_different_seeds() -> None:
    keys = {name: KeyAgreement() for name in NAMES}
    seed = lambda a, b: keys[a].shared_seed(a, b, keys[b].public_key)  # noqa: E731
    pairs = [(NAMES[0], NAMES[1]), (NAMES[0], NAMES[2]), (NAMES[1], NAMES[2])]
    assert all(seed(a, b) == seed(b, a) for a, b in pairs)
    assert len({seed(a, b) for a, b in pairs}) == 3


# Every run makes new keys, so secrets are never reused between trainings.
def test_new_keys_every_time() -> None:
    assert KeyAgreement().public_key != KeyAgreement().public_key


# Public keys that would make the shared secret guessable are rejected.
def test_bad_public_keys_are_rejected() -> None:
    outside_subgroup = next(k for k in range(2, 100) if pow(k, Q, P) == P - 1)
    for bad in (0, 1, P - 1, P, outside_subgroup):
        assert not is_valid_public_key(bad)
        with pytest.raises(ValueError):
            KeyAgreement().shared_seed("A", "B", bad)
    assert is_valid_public_key(KeyAgreement().public_key)


# --- Fixed-point encoding --------------------------------------------------------

# Encode then decode gives the value back within the rounding bound, negatives included.
def test_encode_decode_round_trip() -> None:
    values = np.array([0.36, -0.83, 0.0, -42.4, 476405.0, 1e-9])
    assert np.abs(decode(encode(values)) - values).max() <= 2.0**-33


# Adding encoded values gives the encoded sum; a mask added and removed leaves no trace.
def test_encoded_arithmetic_is_exact() -> None:
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=5) * 40, rng.normal(size=5) * 40
    assert np.abs(decode(encode(a) + encode(b)) - (a + b)).max() < 1e-9
    mask = rng.integers(0, 2**64, size=5, dtype=np.uint64, endpoint=False)
    assert np.array_equal(encode(a) + mask - mask, encode(a))


# Values that could overflow, and NaN or infinity, are refused.
@pytest.mark.parametrize("bad", [MAX_ABS_VALUE, -MAX_ABS_VALUE, np.nan, np.inf])
def test_encode_refuses_unsafe_values(bad: float) -> None:
    with pytest.raises(ValueError):
        encode(np.array([bad]))


# --- Masking ---------------------------------------------------------------------

def hospital_values() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(1)
    return {name: rng.normal(size=16) * 50 for name in NAMES}


# All masks cancel: the server's sum of masked messages is the true sum.
def test_masks_cancel_in_the_sum() -> None:
    maskers, values = make_maskers(NAMES), hospital_values()
    messages = [maskers[n].mask(values[n], "round-1") for n in NAMES]
    assert np.abs(aggregate(messages) - sum(values.values())).max() < 1e-8


# A single masked message looks random: about half of its bits are 1, and decoding it gives nonsense.
def test_single_message_looks_random() -> None:
    maskers, values = make_maskers(NAMES), hospital_values()
    messages = [maskers[n].mask(values[n], f"round-{i}") for i in range(100) for n in NAMES]
    share_of_ones = np.unpackbits(np.concatenate(messages).view(np.uint8)).mean()
    assert 0.49 < share_of_ones < 0.51
    assert np.abs(decode(messages[0]) - values[NAMES[0]]).min() > 1e3


# Each message gets fresh masks: the same values give different messages, and a label cannot be reused.
def test_fresh_masks_and_no_label_reuse() -> None:
    masker, values = make_maskers(NAMES)[NAMES[0]], hospital_values()[NAMES[0]]
    assert not np.array_equal(masker.mask(values, "round-1"), masker.mask(values, "round-2"))
    with pytest.raises(ValueError):
        masker.mask(values, "round-1")


# Server + one hospital: they can strip the masks they share, but another hospital stays hidden;
# they only learn the sum of the other two hospitals.
def test_collusion_with_one_hospital_reveals_only_a_sum() -> None:
    maskers, values = make_maskers(NAMES), hospital_values()
    berlin, chennai, hyderabad = NAMES
    messages = {n: maskers[n].mask(values[n], "round-1") for n in NAMES}
    shared_seed = maskers[berlin]._peer_seeds[chennai]
    attempt = decode(messages[chennai] + _mask(shared_seed, "round-1", 16))  # undo Chennai's "- mask(Berlin, Chennai)"
    assert np.abs(attempt - values[chennai]).min() > 1e3
    total = aggregate(list(messages.values()))
    assert np.allclose(total - values[berlin], values[chennai] + values[hyderabad])


# Known, inherent limit: server + two hospitals recover the third (any sum over three parties has this).
def test_collusion_with_two_hospitals_reveals_the_third() -> None:
    maskers, values = make_maskers(NAMES), hospital_values()
    total = aggregate([maskers[n].mask(values[n], "round-1") for n in NAMES])
    assert np.allclose(total - values[NAMES[0]] - values[NAMES[1]], values[NAMES[2]])


# Known limitation: no dropout recovery. Without one hospital's message the masks do not cancel.
@pytest.mark.xfail(strict=True, reason="no dropout recovery (Bonawitz et al. use secret sharing for this)")
def test_sum_survives_a_missing_hospital() -> None:
    maskers, values = make_maskers(NAMES), hospital_values()
    partial = aggregate([maskers[n].mask(values[n], "round-1") for n in NAMES[:2]])
    assert np.allclose(partial, values[NAMES[0]] + values[NAMES[1]])


# --- Secure aggregation inside FedAvg ---------------------------------------------

# Secure FedAvg gives the same model as plain FedAvg, up to fixed-point rounding.
def test_secure_fedavg_matches_plain() -> None:
    plain, _ = train_fedavg(partition_by_hospital(TRAIN), ROUNDS, LOCAL_EPOCHS, CONFIG, seed=7, secure=False)
    secure, _ = train_fedavg(partition_by_hospital(TRAIN), ROUNDS, LOCAL_EPOCHS, CONFIG, seed=7, secure=True)
    assert np.abs(plain.params - secure.params).max() < 1e-9
    assert np.array_equal(plain.standardizer.mean, secure.standardizer.mean)


class SpyClient(HospitalClient):
    """A real hospital client that records every message it sends to the server."""

    def __init__(self, name, records):
        super().__init__(name, records)
        self.sent = []

    def totals_message(self, *args, **kwargs):
        self.sent.append(super().totals_message(*args, **kwargs))
        return self.sent[-1]

    def update_message(self, *args, **kwargs):
        self.sent.append(super().update_message(*args, **kwargs))
        return self.sent[-1]

    def loss_message(self, *args, **kwargs):
        self.sent.append(super().loss_message(*args, **kwargs))
        return self.sent[-1]


# With secure aggregation on, every message a hospital sends is masked (whole numbers modulo 2^64),
# and no single hospital's case count can be read; only the total of 120 comes out of the sum.
def test_server_only_receives_masked_messages() -> None:
    by_site = {}
    for record in TRAIN:
        by_site.setdefault(record["hospital_id"], []).append(record)
    clients = {name: SpyClient(name, rows) for name, rows in by_site.items()}
    train_fedavg(clients, rounds=3, local_epochs=1, config=CONFIG, seed=7, secure=True)
    for client in clients.values():
        assert len(client.sent) == 1 + 2 * 3
        assert all(message.dtype == np.uint64 for message in client.sent)
        assert abs(decode(client.sent[0])[0] - client.n_cases) > 1e3  # the totals message hides the case count
    assert round(aggregate([c.sent[0] for c in clients.values()])[0]) == 120


# A hospital cannot send masked messages before key agreement, or finish a key agreement it never started.
def test_key_agreement_must_come_first() -> None:
    client = partition_by_hospital(TRAIN)[NAMES[0]]
    with pytest.raises(RuntimeError):
        client.totals_message("feature-totals", masked=True)
    with pytest.raises(RuntimeError):
        client.finish_key_agreement({})


# --- Privacy summary ----------------------------------------------------------------

@pytest.fixture(scope="module")
def evaluation():
    return evaluate_secure_aggregation(TRAIN, ROUNDS, LOCAL_EPOCHS, CONFIG, seed=7)


# The evaluation shows the leak being closed: without secure aggregation the server can order the
# hospitals by readmission rate from round 1 and read Berlin's 0 CKD patients; with it, only totals.
def test_evaluation_shows_leak_and_fix(evaluation) -> None:
    plain = evaluation["server_view_without_secure_aggregation"]
    assert plain["inference_from_round_1_update"]["orders_match"]
    assert plain["per_hospital"]["BERLIN_NODE"]["patients_with_ckd"] == 0
    secure = evaluation["server_view_with_secure_aggregation"]
    assert secure["what_the_sum_reveals"]["training_cases"] == 120
    assert abs(secure["single_message_decoded_as_if_plain"]["BERLIN_NODE"]["patients_with_ckd"]) > 1e3
    assert evaluation["utility"]["largest_weight_difference_plain_vs_secure"] < 1e-9


# privacy_summary.json contains every item SUBMISSION_SCHEMA.md asks for.
def test_privacy_summary_has_required_fields(evaluation) -> None:
    summary = build_privacy_summary(evaluation, ROUNDS)
    for key in (
        "mechanism", "implementation_status", "protected_asset", "adversary", "trust_assumptions", "parameters",
        "privacy_claim", "not_guaranteed", "utility_analysis", "runtime_analysis", "remaining_attack_surface",
        "limitations", "relation_to_federated_learning",
    ):
        assert summary[key], key
    assert any(not a["protected"] for a in summary["adversary"])  # the limits are stated, not only the protections
    assert str(len(FEATURES) + 1) in summary["parameters"]["masked_messages_per_hospital"]
