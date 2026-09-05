from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pvr_pca import run_variance_reduced_pca


VARIANTS = {
    "VR-PCA": ("procrustes", "svd"),
    "Projection-only": ("least_squares", "svd"),
    "QR-only": ("procrustes", "qr"),
    "PVR-PCA": ("least_squares", "qr"),
}


def construct_data(samples: int, features: int, rank: int, gap: float, seed: int):
    rng = np.random.default_rng(seed)
    left, _ = np.linalg.qr(rng.standard_normal((samples, features)), mode="reduced")
    right, _ = np.linalg.qr(rng.standard_normal((features, features)))
    leading = 1 - gap * np.r_[0, np.arange(1, rank) / 10 + 0.9]
    tail = np.abs(rng.standard_normal(features - rank)) / features
    singular_values = np.r_[leading, tail]
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        data = (left * singular_values) @ right.T
    if not np.isfinite(data).all():
        raise FloatingPointError("synthetic data contain non-finite values")
    return data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=20000)
    parser.add_argument("--features", type=int, default=100)
    parser.add_argument("--rank", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--epoch-length", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--gaps", type=float, nargs="+", default=[0.008, 0.016, 0.032])
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    curves = {}
    for gap in args.gaps:
        data = construct_data(args.samples, args.features, args.rank, gap, args.seed)
        mean_norm = float(np.mean(np.sum(data * data, axis=1)))
        step_size = min(1e-3, 1 / (args.rank * mean_norm * np.sqrt(data.shape[0])))
        for variant, (projection, orthogonalization) in VARIANTS.items():
            for repeat in range(args.repeats):
                result = run_variance_reduced_pca(
                    data, args.rank, args.epochs, args.epoch_length, step_size, projection,
                    orthogonalization, args.seed + repeat,
                )
                rows.append({
                    "gap": gap, "variant": variant, "repeat": repeat,
                    "seconds": result.elapsed_seconds,
                    "final_objective": result.objectives[-1],
                })
                curves[f"{gap}:{variant}:{repeat}"] = result.objectives
                print(gap, variant, repeat, f"{result.elapsed_seconds:.3f}s", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "synthetic_benchmark.csv", index=False)
    summary = frame.groupby(["gap", "variant"], sort=False).agg(
        seconds_median=("seconds", "median"),
        objective_median=("final_objective", "median"),
    ).reset_index()
    vr = summary[summary.variant == "VR-PCA"][["gap", "seconds_median"]].rename(
        columns={"seconds_median": "vr_seconds"})
    summary = summary.merge(vr, on="gap")
    summary["speedup_vs_vr"] = summary.vr_seconds / summary.seconds_median
    summary.to_csv(output / "synthetic_summary.csv", index=False)
    for variant in VARIANTS:
        subset = summary[summary.variant == variant]
        plt.plot(subset.gap, subset.seconds_median, "-o", label=variant)
    plt.xlabel("Eigengap")
    plt.ylabel(f"Wall-clock time for {args.epochs} epochs (s)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output / "synthetic_runtime_by_gap.pdf")


if __name__ == "__main__":
    main()
