from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import load_npz
from scipy.io import loadmat
from scipy.sparse.linalg import svds


def covariance_action(data, basis):
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        return data.T @ (data @ basis) / data.shape[0]


def polar_svd(matrix):
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        left, _, right_t = np.linalg.svd(matrix, full_matrices=False)
        return left @ right_t


def cholesky_qr(matrix):
    gram = matrix.T @ matrix
    gram = (gram + gram.T) / 2
    try:
        factor = np.linalg.cholesky(gram).T
        return np.linalg.solve(factor.T, matrix.T).T
    except np.linalg.LinAlgError:
        return np.linalg.qr(matrix, mode="reduced")[0]


def orthogonalize(matrix, method):
    if method == "polar":
        return polar_svd(matrix)
    if method == "chol":
        return cholesky_qr(matrix)
    return np.linalg.qr(matrix, mode="reduced")[0]


def sample_action(data, sample_index, basis):
    if hasattr(data, "indptr"):
        start, end = data.indptr[sample_index:sample_index + 2]
        indices = data.indices[start:end]
        values = data.data[start:end]
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            response = values @ basis[indices]
        return indices, values, response
    values = data[sample_index]
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        response = values @ basis
    return None, values, response


def add_rank_one(candidate, indices, values, response, scale):
    update = scale * values[:, None] * response[None, :]
    if indices is None:
        candidate += update
    else:
        candidate[indices] += update


def oja_warm_start(data, rank, seed, output, passes=4, coefficient=243.0,
                   orth_method="qr", warm_sample_seed=None):
    coefficient_tag = f"{coefficient:g}".replace(".", "p")
    warm_seed_tag = "cont" if warm_sample_seed is None else str(warm_sample_seed)
    cache = output / (
        f"ccat_warmstart_n{data.shape[0]}_d{data.shape[1]}_k{rank}_"
        f"powerqr_oja{passes}_c{coefficient_tag}_"
        f"{orth_method}_seed{seed}_warm{warm_seed_tag}.npy")
    if cache.exists():
        return np.load(cache), True
    rng = np.random.default_rng(seed)
    initial = rng.standard_normal((data.shape[1], rank))
    basis = orthogonalize(covariance_action(data, initial), orth_method)
    sample_seed = seed if warm_sample_seed is None else warm_sample_seed
    sample_cache = output / (
        f"ccat_oja_samples_n{data.shape[0]}_p{passes}_seed{sample_seed}.npy")
    if sample_cache.exists():
        samples = np.load(sample_cache, mmap_mode="r")
    else:
        sample_rng = rng if warm_sample_seed is None else np.random.default_rng(warm_sample_seed)
        samples = sample_rng.integers(0, data.shape[0], size=(passes, data.shape[0]), dtype=np.int32)
        np.save(sample_cache, samples)
    iteration = 0
    for warm_pass in range(passes):
        for sample_index in samples[warm_pass]:
            iteration += 1
            indices, values, response = sample_action(data, sample_index, basis)
            add_rank_one(basis, indices, values, response, coefficient / iteration)
            basis = orthogonalize(basis, orth_method)
        print("Oja warm pass", warm_pass + 1, flush=True)
    np.save(cache, basis)
    return basis, False


def run_method(data, rank, epochs, method, seed, output, step_coefficient=None,
               warm_passes=4, orth_backend="paper", warm_coefficient=243.0,
               sample_seed=None, warm_sample_seed=None, check_finite=False,
               tolerance=None):
    sample_seed = seed if sample_seed is None else sample_seed
    rng = np.random.default_rng(sample_seed)
    warm_orth = "chol" if orth_backend == "chol" else "qr"
    initialization_started = perf_counter()
    snapshot, initialization_cache_hit = oja_warm_start(
        data, rank, seed, output, passes=warm_passes,
        coefficient=warm_coefficient, orth_method=warm_orth,
        warm_sample_seed=warm_sample_seed)
    initialization_seconds = perf_counter() - initialization_started
    mean_norm = float(np.mean(np.asarray(data.multiply(data).sum(axis=1)))) if hasattr(data, "multiply") else float(np.mean(np.sum(data * data, axis=1)))
    step_size = (1 / rank if step_coefficient is None else step_coefficient) / (
        mean_norm * np.sqrt(data.shape[0]))
    singular_values = svds(data, k=rank, return_singular_vectors=False)
    reference = float(np.sum(singular_values ** 2) / data.shape[0])
    errors = []
    epoch_seconds = []
    samples_path = output / (
        f"ccat_samples_n{data.shape[0]}_e{epochs}_seed{sample_seed}.npy")
    if samples_path.exists():
        samples = np.load(samples_path, mmap_mode="r")
    else:
        samples = rng.integers(0, data.shape[0], size=(epochs, data.shape[0]), dtype=np.int32)
        np.save(samples_path, samples)

    basis = snapshot.copy()
    for epoch in range(epochs):
        started = perf_counter()
        full_gradient = covariance_action(data, snapshot)
        for sample_index in samples[epoch]:
            indices, values, basis_response = sample_action(data, sample_index, basis)
            _, _, snapshot_response = sample_action(data, sample_index, snapshot)
            uses_procrustes = method in {"VR-PCA", "QR-only"}
            if uses_procrustes:
                with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                    left, _, right_t = np.linalg.svd(
                        basis.T @ snapshot, full_matrices=False)
                    alignment = right_t.T @ left.T
            else:
                with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                    alignment = snapshot.T @ basis
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                residual = basis_response - snapshot_response @ alignment
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                candidate = basis + step_size * (full_gradient @ alignment)
            add_rank_one(candidate, indices, values, residual, step_size)
            if check_finite and not np.isfinite(candidate).all():
                raise FloatingPointError(
                    f"non-finite candidate in {method}, epoch={epoch + 1}, sample={sample_index}")
            if orth_backend == "paper":
                backend = "polar" if method in {"VR-PCA", "Projection-only"} else "qr"
            else:
                backend = orth_backend
            basis = orthogonalize(candidate, backend)
            if check_finite and not np.isfinite(basis).all():
                raise FloatingPointError(
                    f"non-finite orthogonal basis in {method}, epoch={epoch + 1}, sample={sample_index}")
        snapshot = basis
        objective = float(np.sum(snapshot * covariance_action(data, snapshot)))
        errors.append(max(1 - objective / reference, 1e-16))
        epoch_seconds.append(perf_counter() - started)
        state = {"method": method, "rank": rank, "errors": errors,
                 "epoch_seconds": epoch_seconds,
                 "initialization_seconds": initialization_seconds,
                 "initialization_cache_hit": initialization_cache_hit,
                 "main_seconds": sum(epoch_seconds),
                 "total_seconds": initialization_seconds + sum(epoch_seconds),
                 "step_size": step_size, "step_coefficient": step_coefficient,
                 "samples": data.shape[0], "features": data.shape[1],
                 "seed": seed, "sample_seed": sample_seed,
                 "warm_sample_seed": warm_sample_seed,
                 "warm_passes": warm_passes,
                 "warm_coefficient": warm_coefficient,
                 "stopping_tolerance": tolerance,
                 "finite_check_enabled": check_finite,
                 "alignment": "procrustes" if uses_procrustes else "least_squares",
                 "orthogonalization": backend,
                 "orthogonalization_protocol": orth_backend}
        (output / f"ccat_original_{method.lower()}.json").write_text(json.dumps(state, indent=2))
        print(method, epoch + 1, errors[-1], epoch_seconds[-1], flush=True)
        if tolerance is not None and errors[-1] < tolerance:
            break


def run_oja(data, rank, epochs, coefficient, seed, output, warm_passes=4,
            orth_backend="qr", warm_coefficient=243.0, sample_seed=None):
    sample_seed = seed if sample_seed is None else sample_seed
    snapshot, _ = oja_warm_start(
        data, rank, seed, output, passes=warm_passes,
        coefficient=warm_coefficient, orth_method=orth_backend)
    singular_values = svds(data, k=rank, return_singular_vectors=False)
    reference = float(np.sum(singular_values ** 2) / data.shape[0])
    samples_path = output / (
        f"ccat_samples_n{data.shape[0]}_e{epochs}_seed{sample_seed}.npy")
    if samples_path.exists():
        samples = np.load(samples_path, mmap_mode="r")
    else:
        rng = np.random.default_rng(sample_seed)
        samples = rng.integers(0, data.shape[0], size=(epochs, data.shape[0]), dtype=np.int32)
        np.save(samples_path, samples)
    errors, epoch_seconds = [], []
    iteration = 0
    for epoch in range(epochs):
        started = perf_counter()
        for sample_index in samples[epoch]:
            iteration += 1
            indices, values, response = sample_action(data, sample_index, snapshot)
            add_rank_one(snapshot, indices, values, response, coefficient / iteration)
            snapshot = orthogonalize(snapshot, orth_backend)
        objective = float(np.sum(snapshot * covariance_action(data, snapshot)))
        errors.append(max(1 - objective / reference, 1e-16))
        epoch_seconds.append(perf_counter() - started)
        state = {"method": f"Oja-{int(coefficient)}", "rank": rank,
                 "errors": errors, "epoch_seconds": epoch_seconds}
        (output / f"ccat_original_oja-{int(coefficient)}.json").write_text(
            json.dumps(state, indent=2))
        print("Oja", coefficient, epoch + 1, errors[-1], epoch_seconds[-1], flush=True)


def plot_results(output):
    figure, axis = plt.subplots(figsize=(5.2, 3.8))
    styles = [("PVR-PCA", "#1f77b4", "-"), ("Projection-only", "#2ca02c", "-."),
              ("QR-only", "#9467bd", ":"), ("VR-PCA", "#d95f02", "-"),
              ("Oja-9", "#e6ab02", "--"), ("Oja-81", "#7570b3", "--"),
              ("Oja-243", "#66a61e", "--")]
    for method, color, line_style in styles:
        state_path = output / f"ccat_original_{method.lower()}.json"
        if not state_path.exists():
            continue
        state = json.loads(state_path.read_text())
        axis.semilogy(np.arange(1, len(state["errors"]) + 1),
                  state["errors"], label=method, color=color,
                  linestyle=line_style, linewidth=1.8)
    axis.set_xlabel("Epoch")
    axis.set_ylabel("error")
    axis.grid(alpha=.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output / "ccat_original_protocol.pdf")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument(
        "--method",
        choices=["VR-PCA", "PVR-PCA", "Projection-only", "QR-only", "both", "four", "all"],
        default="both")
    parser.add_argument("--step-coefficient", type=float)
    parser.add_argument("--warm-passes", type=int, default=0)
    parser.add_argument("--warm-coefficient", type=float, default=243.0)
    parser.add_argument("--orth-backend", choices=["paper", "polar", "qr", "chol"], default="paper")
    parser.add_argument("--mat-variable")
    parser.add_argument("--pad-features", type=int, default=0)
    parser.add_argument("--independent-samples", action="store_true")
    parser.add_argument("--independent-initializations", action="store_true")
    parser.add_argument("--independent-warm-samples", action="store_true")
    parser.add_argument("--check-finite", action="store_true")
    parser.add_argument("--tolerance", type=float)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.data.suffix == ".npz":
        data = load_npz(args.data).tocsr()
    elif args.data.suffix == ".npy":
        data = np.asarray(np.load(args.data), dtype=np.float64, order="C")
    else:
        if not args.mat_variable:
            parser.error("--mat-variable is required for MATLAB input")
        data = np.asarray(loadmat(args.data, variable_names=[args.mat_variable])[args.mat_variable].T,
                          dtype=np.float64, order="C")
        if args.pad_features:
            if args.pad_features < data.shape[1]:
                parser.error("--pad-features cannot be smaller than the input dimension")
            data = np.pad(data, ((0, 0), (0, args.pad_features - data.shape[1])))
    if args.method == "both":
        methods = ["PVR-PCA", "VR-PCA"]
    elif args.method in {"four", "all"}:
        methods = ["PVR-PCA", "Projection-only", "QR-only", "VR-PCA"]
    else:
        methods = [args.method]
    for method_index, method in enumerate(methods):
        initialization_seed = args.seed + method_index if args.independent_initializations else args.seed
        sample_seed = args.seed + 1000 + method_index if args.independent_samples else args.seed
        warm_sample_seed = args.seed + 2000 + method_index if args.independent_warm_samples else None
        run_method(data, args.rank, args.epochs, method, initialization_seed, args.output,
                   args.step_coefficient, args.warm_passes, args.orth_backend,
                   args.warm_coefficient, sample_seed, warm_sample_seed,
                   args.check_finite, args.tolerance)
    if args.method == "all":
        for oja_index, coefficient in enumerate([9.0, 81.0, 243.0], start=2):
            backend = "chol" if args.orth_backend == "chol" else "qr"
            sample_seed = args.seed + oja_index if args.independent_samples else args.seed
            run_oja(data, args.rank, args.epochs, coefficient, args.seed, args.output,
                    args.warm_passes, backend, args.warm_coefficient, sample_seed)
    plot_results(args.output)


if __name__ == "__main__":
    main()
