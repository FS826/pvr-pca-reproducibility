from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.utils import check_random_state
from scipy.io import loadmat
from scipy.sparse import load_npz

from pvr_pca import run_variance_reduced_pca


VARIANTS = {
    "VR-PCA": ("procrustes", "svd"),
    "Projection-only": ("least_squares", "svd"),
    "QR-only": ("procrustes", "qr"),
    "PVR-PCA": ("least_squares", "qr"),
}


def load_data(name: str, source: Path, samples: int, seed: int):
    if name == "mnist":
        if source.suffix == ".npy":
            data = np.asarray(np.load(source), dtype=np.float64, order="C")
        else:
            data = loadmat(source, variable_names=["train2_mnist"])["train2_mnist"].T
            data = np.asarray(data, dtype=np.float64)
            data -= data.mean(axis=0, keepdims=True)
    else:
        if source.suffix == ".npz":
            data = load_npz(source).astype(np.float64).tocsr()
        else:
            data = loadmat(source, variable_names=["ccat_training_data"])["ccat_training_data"]
            data = data.astype(np.float64).tocsr()
    if 0 < samples < data.shape[0]:
        indices = check_random_state(seed).choice(data.shape[0], samples, replace=False)
        data = data[indices]
    return data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["mnist", "ccat"], default="mnist")
    parser.add_argument("--samples", type=int, default=0)
    parser.add_argument("--ranks", type=int, nargs="+", default=[3, 6, 17, 28, 64])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--epoch-length", type=int, default=1000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--output", type=Path, default=Path("../output/benchmarks"))
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    data = load_data(args.dataset, args.source, args.samples, args.seed)
    mean_norm = float(np.asarray(data.multiply(data).sum(axis=1) if hasattr(data, "multiply") else (data * data).sum(axis=1)).mean())
    benchmark_path = args.output / f"{args.dataset}_benchmark.csv"
    curves_path = args.output / f"{args.dataset}_curves.json"
    rows = pd.read_csv(benchmark_path).to_dict("records") if benchmark_path.exists() else []
    curves = json.loads(curves_path.read_text()) if curves_path.exists() else {}
    completed = {(row["rank"], row["variant"], row["repeat"]) for row in rows}
    for rank in args.ranks:
        step_size = min(1e-3, 1.0 / max(rank * mean_norm * np.sqrt(data.shape[0]), 1.0))
        for variant, (projection, orthogonalization) in VARIANTS.items():
            for repeat in range(args.repeats):
                if (rank, variant, repeat) in completed:
                    continue
                result = run_variance_reduced_pca(
                    data, rank, args.epochs, args.epoch_length, step_size,
                    projection, orthogonalization, args.seed + repeat,
                )
                rows.append({
                    "dataset": args.dataset, "samples": data.shape[0], "features": data.shape[1],
                    "rank": rank, "variant": variant, "repeat": repeat,
                    "seconds": result.elapsed_seconds,
                    "seconds_per_epoch": np.mean(result.epoch_seconds),
                    "final_objective": result.objectives[-1],
                })
                curves[f"{rank}:{variant}:{repeat}"] = result.objectives
                print(args.dataset, rank, variant, repeat, f"{result.elapsed_seconds:.3f}s", flush=True)
                pd.DataFrame(rows).to_csv(benchmark_path, index=False)
                curves_path.write_text(json.dumps(curves, indent=2))
    frame = pd.DataFrame(rows)
    summary = frame.groupby(["rank", "variant"], sort=False).agg(
        seconds_median=("seconds", "median"),
        seconds_iqr=("seconds", lambda values: values.quantile(.75) - values.quantile(.25)),
        objective_median=("final_objective", "median"),
    ).reset_index()
    vr_time = summary[summary.variant == "VR-PCA"][["rank", "seconds_median"]].rename(
        columns={"seconds_median": "vr_seconds"})
    summary = summary.merge(vr_time, on="rank")
    summary["speedup_vs_vr"] = summary.vr_seconds / summary.seconds_median
    summary.to_csv(args.output / f"{args.dataset}_summary.csv", index=False)
    for variant in VARIANTS:
        subset = summary[summary.variant == variant]
        plt.plot(subset["rank"], subset["seconds_median"], marker="o", label=variant)
    plt.ylabel("Wall-clock time (seconds)")
    plt.xlabel("Target rank k")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output / f"{args.dataset}_runtime_by_k.pdf")


if __name__ == "__main__":
    main()
