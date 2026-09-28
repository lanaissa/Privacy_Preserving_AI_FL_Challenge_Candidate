"""Secure aggregation: the server learns only the sum of the hospitals' numbers.

Step 1 of the protocol is pairwise key agreement. Every pair of hospitals derives a
shared secret seed with Diffie-Hellman; the server relays the public keys but cannot
compute the seeds. The seeds are later used to generate masks that cancel in the sum.
Prototype only: public keys are assumed to arrive unchanged (authenticated channels).
"""
from __future__ import annotations

import hashlib
import secrets

import numpy as np

# RFC 3526, group 14: the 2048-bit MODP group. P is a safe prime (Q = (P - 1) / 2 is
# prime) and G = 2 generates the subgroup of order Q.
P = int(
    "FFFFFFFFFFFFFFFFC90FDAA22168C234C4C6628B80DC1CD129024E088A67CC74020BBEA63B139B22514A0879"
    "8E3404DDEF9519B3CD3A431B302B0A6DF25F14374FE1356D6D51C245E485B576625E7EC6F44C42E9A637ED6B"
    "0BFF5CB6F406B7EDEE386BFB5A899FA5AE9F24117C4B1FE649286651ECE45B3DC2007CB8A163BF0598DA4836"
    "1C55D39A69163FA8FD24CF5F83655D23DCA3AD961C62F356208552BB9ED529077096966D670C354E4ABC9804"
    "F1746C08CA18217C32905E462E36CE3BE39E772C180E86039B2783A2EC07A28FB5C55DF06F4C52C9DE2BCBF6"
    "955817183995497CEA956AE515D2261898FA051015728E5A8AACAA68FFFFFFFFFFFFFFFF",
    16,
)
G = 2
Q = (P - 1) // 2
_KEY_BYTES = (P.bit_length() + 7) // 8
_CONTEXT = b"fl-readmission-secure-aggregation-v1"


def is_valid_public_key(key: int) -> bool:
    """Reject keys that would make the shared secret guessable (0, 1, P-1, or outside the subgroup)."""
    return 2 <= key <= P - 2 and pow(key, Q, P) == 1


class KeyAgreement:
    """One hospital's Diffie-Hellman key pair. The private key never leaves this object."""

    def __init__(self) -> None:
        # Drawn from the operating system's secure random generator.
        self._private_key = secrets.randbelow(Q - 2) + 2
        self.public_key = pow(G, self._private_key, P)

    def shared_seed(self, own_name: str, peer_name: str, peer_public_key: int) -> bytes:
        """32-byte seed shared with one other hospital; both sides compute the same value."""
        if not is_valid_public_key(peer_public_key):
            raise ValueError(f"invalid public key received from {peer_name}")
        shared_secret = pow(peer_public_key, self._private_key, P)
        # Hash the secret together with the pair's names, so each pair gets its own seed.
        pair = "|".join(sorted([own_name, peer_name])).encode()
        return hashlib.sha256(_CONTEXT + shared_secret.to_bytes(_KEY_BYTES, "big") + pair).digest()


# --- Fixed-point encoding ------------------------------------------------------
# Masks only cancel exactly with whole numbers, so decimals are scaled by 2^32 and
# rounded, and all arithmetic is done modulo 2^64 (numpy uint64 wraps around).
# Negative numbers use two's complement, like ordinary 64-bit integers.

FRACTION_BITS = 32
SCALE = 2**FRACTION_BITS
# Each value must stay below 2^29 (about 537 million) so that adding up to 4 of them
# still fits in a signed 64-bit number. Our largest value is about 476,000.
MAX_ABS_VALUE = 2 ** (63 - FRACTION_BITS - 2)


def encode(values: np.ndarray) -> np.ndarray:
    """Decimals -> whole numbers modulo 2^64. Rounding error is at most 2^-33 per value."""
    values = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(values)) or np.any(np.abs(values) >= MAX_ABS_VALUE):
        raise ValueError("value is not finite or too large to encode safely")
    return np.round(values * SCALE).astype(np.int64).view(np.uint64)


def decode(encoded: np.ndarray) -> np.ndarray:
    """Whole numbers modulo 2^64 -> decimals (values above 2^63 are read as negative)."""
    return np.asarray(encoded, dtype=np.uint64).view(np.int64).astype(float) / SCALE


# --- Pairwise masking ----------------------------------------------------------

def _mask(seed: bytes, label: str, length: int) -> np.ndarray:
    """Random-looking whole numbers from a pair's secret seed and a one-time label (SHAKE-256)."""
    stream = hashlib.shake_256(_CONTEXT + seed + label.encode()).digest(8 * length)
    return np.frombuffer(stream, dtype="<u8").astype(np.uint64)


class PairwiseMasker:
    """Hides one hospital's numbers under masks that cancel when all hospitals are added up.

    For each pair of hospitals, the one whose name sorts first adds the pair's mask and
    the other subtracts it, so every mask appears once with + and once with - in the sum.
    """

    def __init__(self, name: str, peer_seeds: dict[str, bytes]) -> None:
        self.name = name
        self._peer_seeds = dict(peer_seeds)  # private: the seeds never leave this object
        self._used_labels: set[str] = set()

    def mask(self, values: np.ndarray, label: str) -> np.ndarray:
        """The message sent to the server: encoded values plus this hospital's masks."""
        # A reused label would reuse the masks, and subtracting two messages would cancel them.
        if label in self._used_labels:
            raise ValueError(f"{self.name}: mask label {label!r} was already used")
        self._used_labels.add(label)
        message = encode(values)
        for peer, seed in sorted(self._peer_seeds.items()):
            mask = _mask(seed, label, len(message))
            message = message + mask if self.name < peer else message - mask
        return message


def aggregate(messages: list[np.ndarray]) -> np.ndarray:
    """Server side: add all hospitals' messages; the masks cancel and only the total remains.

    Needs a message from every hospital: if one is missing, its masks do not cancel.
    """
    total = np.zeros_like(messages[0], dtype=np.uint64)
    for message in messages:
        total = total + message
    return decode(total)
