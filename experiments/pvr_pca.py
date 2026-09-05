from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy import sparse


def orthonormalize(matrix: np.ndarray, method: str) -> np.ndarray:
    if method == "qr":
        return np.linalg.qr(matrix, mode="reduced")[0]
    left, _, right_t = np.linalg.svd(matrix, full_matrices=False)
    return left @ right_t


def covariance_action(data, basis: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        result = data.T @ (data @ basis) / data.shape[0]
    if not np.isfinite(result).all():
        raise FloatingPointError("non-finite covariance action")
    return result


def warm_start(data, rank: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    basis = rng.standard_normal((data.shape[1], rank), dtype=np.float64)
    return orthonormalize(covariance_action(data, basis), "qr")


@dataclass
class RunResult:
    basis: np.ndarray
    elapsed_seconds: float
    epoch_seconds: list[float]
    objectives: list[float]


def run_variance_reduced_pca(
    data,
    rank: int,
    epochs: int,
    epoch_length: int,
    step_size: float,
    projection: str,
    orthogonalization: str,
    seed: int,
) -> RunResult:
    rng = np.random.default_rng(seed)
    snapshot = warm_start(data, rank, seed)
    objectives = [float(np.sum(snapshot * covariance_action(data, snapshot)))]
    epoch_seconds: list[float] = []
    started = perf_counter()

    for _ in range(epochs):
        epoch_started = perf_counter()
        full_gradient = covariance_action(data, snapshot)
        basis = snapshot.copy()
        for index in rng.integers(0, data.shape[0], size=epoch_length):
            sample = data[index]
            row = sample.toarray().ravel() if sparse.issparse(data) else np.asarray(sample)
            if projection == "least_squares":
                alignment = snapshot.T @ basis
            else:
                left, _, right_t = np.linalg.svd(basis.T @ snapshot, full_matrices=False)
                alignment = right_t.T @ left.T
            residual = row @ basis - (row @ snapshot) @ alignment
            update = np.outer(row, residual) + full_gradient @ alignment
            basis = orthonormalize(basis + step_size * update, orthogonalization)
        snapshot = basis
        epoch_seconds.append(perf_counter() - epoch_started)
        objectives.append(float(np.sum(snapshot * covariance_action(data, snapshot))))

    return RunResult(snapshot, perf_counter() - started, epoch_seconds, objectives)
