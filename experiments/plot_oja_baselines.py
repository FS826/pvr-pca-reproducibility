from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METHODS = ["PVR-PCA", "VR-PCA", "Oja-9", "Oja-81", "Oja-243"]
STYLES = {
    "PVR-PCA": ("#1f77b4", "-"),
    "VR-PCA": ("#d95f02", "--"),
    "Oja-9": ("#e6ab02", ":"),
    "Oja-81": ("#7570b3", "-."),
    "Oja-243": ("#2ca02c", ":"),
}


def state(directory: Path, method: str) -> dict:
    return json.loads((directory / f"ccat_original_{method.lower()}.json").read_text())


def plot_mnist() -> None:
    figure, axes = plt.subplots(1, 3, figsize=(13.2, 3.7))
    for axis, rank in zip(axes, [3, 17, 28]):
        directory = Path(f"output/mnist_original_python/k{rank}")
        for method in METHODS:
            result = state(directory, method)
            elapsed = np.cumsum(result["epoch_seconds"])
            color, line_style = STYLES[method]
            axis.semilogy(elapsed, result["errors"], color=color,
                          linestyle=line_style, linewidth=1.6, label=method)
        axis.set_title(f"MNIST, k={rank}")
        axis.set_xlabel("Cumulative wall-clock time (s)")
        axis.set_ylabel("Relative subspace error")
        axis.grid(alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=5, frameon=False)
    figure.tight_layout(rect=(0, 0, 1, 0.88))
    output = Path("output/oja_baselines")
    output.mkdir(parents=True, exist_ok=True)
    figure.savefig(output / "MNIST_Oja_baselines.pdf")
    figure.savefig(output / "MNIST_Oja_baselines.png", dpi=220)


def plot_ccat() -> None:
    directory = Path("output/ccat_python/subset_eta3_protocol")
    figure, axis = plt.subplots(figsize=(5.3, 3.8))
    for method in METHODS:
        result = state(directory, method)
        elapsed = np.cumsum(result["epoch_seconds"])
        color, line_style = STYLES[method]
        axis.semilogy(elapsed, result["errors"], color=color,
                      linestyle=line_style, linewidth=1.6, label=method)
    axis.set_title("CCAT subset, k=6")
    axis.set_xlabel("Cumulative wall-clock time (s)")
    axis.set_ylabel("Relative subspace error")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()
    output = Path("output/oja_baselines")
    output.mkdir(parents=True, exist_ok=True)
    figure.savefig(output / "CCAT_Oja_baselines.pdf")
    figure.savefig(output / "CCAT_Oja_baselines.png", dpi=220)


if __name__ == "__main__":
    plot_mnist()
    plot_ccat()
