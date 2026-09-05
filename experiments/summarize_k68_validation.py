from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "output" / "mnist_k68_c3_validation"
OUTPUT_ROOT = ROOT / "output" / "mnist_k68_validation_summary"
TOLERANCE = 1e-4


def first_hit(result: dict) -> tuple[int | None, float | None]:
    for index, error in enumerate(result["errors"]):
        if error < TOLERANCE:
            return index + 1, float(sum(result["epoch_seconds"][:index + 1]))
    return None, None


def main() -> None:
    rows = []
    error_runs = {"PVR-PCA": [], "VR-PCA": []}
    for run in sorted(RESULT_ROOT.glob("k68_seed*")):
        for method in ("pvr-pca", "vr-pca"):
            result = json.loads((run / f"ccat_original_{method}.json").read_text())
            label = method.upper()
            error_runs[label].append(result["errors"])
            epoch, seconds = first_hit(result)
            rows.append({
                "seed": result["seed"],
                "method": label,
                "success": epoch is not None,
                "epoch": epoch,
                "seconds": seconds,
                "final_error": result["errors"][-1],
            })

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_ROOT / "MNIST_k68_validation_runs.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    figure, axes = plt.subplots(1, 2, figsize=(8.4, 3.5))
    colors = {"PVR-PCA": "#1f77b4", "VR-PCA": "#d95f02"}
    for position, method in enumerate(("PVR-PCA", "VR-PCA")):
        selected = [row for row in rows if row["method"] == method and row["success"]]
        epochs = np.asarray([row["epoch"] for row in selected], dtype=float)
        seconds = np.asarray([row["seconds"] for row in selected], dtype=float)
        axes[0].bar(position, epochs.mean(), yerr=epochs.std(ddof=1), capsize=4,
                    color=colors[method], alpha=0.75)
        axes[1].bar(position, seconds.mean(), yerr=seconds.std(ddof=1), capsize=4,
                    color=colors[method], alpha=0.75)
    for axis in axes:
        axis.set_xticks((0, 1), ("PVR-PCA", "VR-PCA"))
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel(r"Epochs to $10^{-4}$")
    axes[0].set_title("Iteration count (4/5 successes each)")
    axes[1].set_ylabel(r"Time to $10^{-4}$ (s)")
    axes[1].set_title("Wall-clock time")
    figure.tight_layout()
    figure.savefig(OUTPUT_ROOT / "MNIST_k68_time_to_tolerance.pdf", bbox_inches="tight")
    figure.savefig(OUTPUT_ROOT / "MNIST_k68_time_to_tolerance.png", dpi=220,
                   bbox_inches="tight")
    plt.close(figure)

    convergence, axis = plt.subplots(figsize=(4.8, 3.6))
    colors = {"PVR-PCA": "#1f77b4", "VR-PCA": "#d95f02"}
    for method in ("PVR-PCA", "VR-PCA"):
        runs = error_runs[method]
        common_epochs = min(len(run) for run in runs)
        logged = np.log10(np.maximum(
            np.asarray([run[:common_epochs] for run in runs], dtype=float),
            np.finfo(float).tiny,
        ))
        mean = logged.mean(axis=0)
        standard_deviation = logged.std(axis=0, ddof=1)
        epochs = np.arange(1, common_epochs + 1)
        axis.plot(epochs, mean, marker="o", markersize=3.2, linewidth=1.7,
                  color=colors[method], label=method)
        axis.fill_between(epochs, mean - standard_deviation,
                          mean + standard_deviation, color=colors[method], alpha=0.16)
    axis.axhline(np.log10(TOLERANCE), color="black", linestyle="--", linewidth=1,
                 label=r"Tolerance $10^{-4}$")
    axis.set_xlabel("Epoch")
    axis.set_ylabel(r"Mean $\log_{10}$(relative error)")
    axis.set_title("MNIST, k=68 (five validation seeds)")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    convergence.tight_layout()
    convergence.savefig(OUTPUT_ROOT / "MNIST_k68_epoch_convergence.pdf",
                         bbox_inches="tight")
    convergence.savefig(OUTPUT_ROOT / "MNIST_k68_epoch_convergence.png", dpi=220,
                         bbox_inches="tight")
    plt.close(convergence)


if __name__ == "__main__":
    main()
