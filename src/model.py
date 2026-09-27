"""Logistic regression in numpy, shared by the local, federated and centralized models.

Parameters are one array: params[0] is the bias, params[1:] are the feature weights.
Using the same code for all three models keeps the comparison fair.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TrainingConfig:
    learning_rate: float = 0.1
    l2: float = 0.01  # penalty on large weights (not on the bias)
    batch_size: int = 16


def init_params(n_features: int) -> np.ndarray:
    """All-zero start: every patient gets probability 0.5 before training."""
    return np.zeros(n_features + 1)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    # Clipping avoids overflow in exp() for very large scores; the result is unchanged at float precision.
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35.0, 35.0)))


def predict_proba(params: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Probability of readmission for each row of x."""
    return _sigmoid(params[0] + x @ params[1:])


def loss(params: np.ndarray, x: np.ndarray, y: np.ndarray, l2: float) -> float:
    """Average log loss plus the L2 penalty: the number training tries to make small."""
    p = np.clip(predict_proba(params, x), 1e-12, 1 - 1e-12)
    log_loss = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    return float(log_loss + 0.5 * l2 * np.sum(params[1:] ** 2))


def gradient(params: np.ndarray, x: np.ndarray, y: np.ndarray, l2: float) -> np.ndarray:
    """Direction in which the loss grows fastest; training steps the opposite way."""
    error = predict_proba(params, x) - y
    grad = np.empty_like(params)
    grad[0] = error.mean()
    grad[1:] = x.T @ error / len(y) + l2 * params[1:]
    return grad


def train_epochs(
    params: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    epochs: int,
    config: TrainingConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Mini-batch gradient descent for a number of passes (epochs) over the data.

    Returns new parameters; the input array is not changed. The seeded rng decides
    the shuffling order, which is the only source of randomness.
    """
    params = params.copy()
    for _ in range(epochs):
        order = rng.permutation(len(y))
        for start in range(0, len(y), config.batch_size):
            batch = order[start : start + config.batch_size]
            params -= config.learning_rate * gradient(params, x[batch], y[batch], config.l2)
    return params
