from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "output" / "mnist_historical_k28"
OUTPUT_ROOT = ROOT / "output" / "mnist_k28_time_to_tolerance"
TOLERANCES = (1e-4, 1e-5, 1e-6)


def first_hit(result: dict, tolerance: float) -> tuple[int | None, float | None]:
    for index, error in enumerate(result["errors"]):
        if error < tolerance:
            return index + 1, float(sum(result["epoch_seconds"][:index + 1]))
    return None, None


def main() -> None:
    rows = []
    for run in sorted(RESULT_ROOT.glob("k28_shared_oja_seed*")):
        results = {
            method: json.loads((run / f"ccat_original_{method}.json").read_text())
            for method in ("pvr-pca", "vr-pca")
        }
        for tolerance in TOLERANCES:
            for method, result in results.items():
                epoch, seconds = first_hit(result, tolerance)
                rows.append({
                    "seed": result["seed"],
                    "method": method.upper(),
                    "tolerance": tolerance,
                    "hit": epoch is not None,
                    "epoch": epoch,
                    "seconds": seconds,
                })

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_ROOT / "MNIST_k28_time_to_tolerance_runs.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    summary = []
    for tolerance in TOLERANCES:
        for method in ("PVR-PCA", "VR-PCA"):
            selected = [row for row in rows
                        if row["method"] == method and row["tolerance"] == tolerance]
            epochs = np.asarray([row["epoch"] for row in selected if row["hit"]], dtype=float)
            seconds = np.asarray([row["seconds"] for row in selected if row["hit"]], dtype=float)
            summary.append({
                "method": method,
                "tolerance": tolerance,
                "successes": int(len(seconds)),
                "runs": int(len(selected)),
                "epoch_mean": float(epochs.mean()),
                "epoch_sample_sd": float(epochs.std(ddof=1)),
                "seconds_mean": float(seconds.mean()),
                "seconds_sample_sd": float(seconds.std(ddof=1)),
            })

    with (OUTPUT_ROOT / "MNIST_k28_time_to_tolerance_summary.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)

    figure, axes = plt.subplots(1, 2, figsize=(8.8, 3.5))
    positions = np.arange(len(TOLERANCES))
    width = 0.34
    for offset, method, color in (
        (-width / 2, "PVR-PCA", "#1f77b4"),
        (width / 2, "VR-PCA", "#d95f02"),
    ):
        selected = [row for row in summary if row["method"] == method]
        axes[0].bar(positions + offset, [row["epoch_mean"] for row in selected], width,
                    yerr=[row["epoch_sample_sd"] for row in selected], capsize=3,
                    color=color, alpha=0.75, label=method)
        axes[1].bar(positions + offset, [row["seconds_mean"] for row in selected], width,
                    yerr=[row["seconds_sample_sd"] for row in selected], capsize=3,
                    color=color, alpha=0.75, label=method)
    labels = [r"$10^{-4}$", r"$10^{-5}$", r"$10^{-6}$"]
    for axis in axes:
        axis.set_xticks(positions, labels)
        axis.set_xlabel("Relative-error tolerance")
        axis.grid(axis="y", alpha=0.25)
        axis.legend(frameon=False)
    axes[0].set_ylabel("Epochs to tolerance")
    axes[0].set_title("Iteration count")
    axes[1].set_ylabel("Wall-clock time to tolerance (s)")
    axes[1].set_title("Time to tolerance")
    figure.tight_layout()
    figure.savefig(OUTPUT_ROOT / "MNIST_k28_time_to_tolerance.pdf", bbox_inches="tight")
    figure.savefig(OUTPUT_ROOT / "MNIST_k28_time_to_tolerance.png", dpi=220,
                   bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
