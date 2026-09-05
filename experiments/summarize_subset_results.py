from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_result(folder: Path, method: str) -> dict:
    return json.loads((folder / f"ccat_original_{method.lower()}.json").read_text())


def first_threshold_time(result: dict, threshold: float) -> float:
    elapsed = np.cumsum(result["epoch_seconds"])
    for error, seconds in zip(result["errors"], elapsed):
        if error <= threshold:
            return float(seconds)
    return float("nan")


def summarize(dataset: str, folder: Path, output: Path) -> None:
    results = {method: load_result(folder, method) for method in ("PVR-PCA", "VR-PCA")}
    rows = []
    for method, result in results.items():
        row = {
            "dataset": dataset,
            "method": method,
            "samples": result["samples"],
            "features": result["features"],
            "rank": result["rank"],
            "final_error": result["errors"][-1],
            "mean_epoch_seconds": float(np.mean(result["epoch_seconds"])),
            "main_seconds": result["main_seconds"],
            "total_seconds": result["total_seconds"],
        }
        for threshold in (1e-2, 1e-3, 1e-4, 1e-5, 1e-6):
            row[f"seconds_to_{threshold:g}"] = first_threshold_time(result, threshold)
        rows.append(row)
    frame = pd.DataFrame(rows)
    pvr_seconds = frame.loc[frame.method == "PVR-PCA", "mean_epoch_seconds"].iloc[0]
    vr_seconds = frame.loc[frame.method == "VR-PCA", "mean_epoch_seconds"].iloc[0]
    frame["per_epoch_speedup_vs_vr"] = vr_seconds / frame["mean_epoch_seconds"]
    frame.to_csv(output / f"{dataset}_summary.csv", index=False)

    figure, axes = plt.subplots(1, 2, figsize=(9.2, 3.7))
    colors = {"PVR-PCA": "#d62728", "VR-PCA": "#1f77b4"}
    for method, result in results.items():
        errors = np.maximum(np.asarray(result["errors"]), 1e-16)
        epochs = np.arange(1, len(errors) + 1)
        seconds = np.cumsum(result["epoch_seconds"])
        axes[0].plot(epochs, np.log10(errors), "-o", label=method,
                     color=colors[method], markersize=3)
        axes[1].plot(seconds, np.log10(errors), "-o", label=method,
                     color=colors[method], markersize=3)
    axes[0].set_xlabel("Epoch")
    axes[1].set_xlabel("Main-loop wall-clock time (s)")
    for axis in axes:
        axis.set_ylabel(r"$\log_{10}$ relative objective error")
        axis.grid(alpha=0.25)
        axis.legend()
    figure.suptitle(dataset)
    figure.tight_layout()
    figure.savefig(output / f"{dataset}_error_epoch_and_time.pdf")
    figure.savefig(output / f"{dataset}_error_epoch_and_time.png", dpi=180)
    plt.close(figure)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output = root / "output" / "subset_shamir_protocol" / "summary"
    output.mkdir(parents=True, exist_ok=True)
    summarize("MNIST, 60000x780, k=6",
              root / "output" / "subset_shamir_protocol" / "mnist_k6", output)
    summarize("CCAT subset, 80000x47236, k=3",
              root / "output" / "subset_shamir_protocol" / "ccat_k3", output)


if __name__ == "__main__":
    main()
