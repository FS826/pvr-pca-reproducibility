from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load(folder: Path, method: str) -> np.ndarray:
    path = folder / f"ccat_original_{method.lower()}.json"
    return np.asarray(json.loads(path.read_text())["errors"], dtype=float)


def plot_rank(rank: int) -> None:
    folder = ROOT / "output" / "final_mnist_same_init_independent" / f"k{rank}"
    pvr = np.log10(np.maximum(load(folder, "pvr-pca"), 1e-16))
    vr = np.log10(np.maximum(load(folder, "vr-pca"), 1e-16))
    epochs = np.arange(1, len(pvr) + 1)

    figure, axis = plt.subplots(figsize=(4.6, 3.5))
    axis.plot(epochs, vr, "--s", color="#d95f02", linewidth=1.7,
              markersize=3.8, markerfacecolor="white", markevery=3,
              label="VR-PCA")
    axis.plot(epochs, pvr, "-o", color="#1f77b4", linewidth=1.7,
              markersize=3.6, markevery=3, label="PVR-PCA")
    axis.set_xlabel("Epoch")
    axis.set_ylabel(r"$\log_{10}(\mathrm{relative\ subspace\ error})$")
    axis.set_title(f"MNIST, k={rank}")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(ROOT / "revised" / f"MNIST_{rank}.pdf", bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    for rank in (3, 17, 28):
        plot_rank(rank)


if __name__ == "__main__":
    main()
