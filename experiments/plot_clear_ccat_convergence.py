from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_errors(folder: Path, method: str) -> np.ndarray:
    path = folder / f"ccat_original_{method.lower()}.json"
    return np.asarray(json.loads(path.read_text())["errors"], dtype=float)


def plot_pair(folder: Path, output: Path, title: str, show_gap: bool = False) -> None:
    pvr = np.log10(np.maximum(load_errors(folder, "pvr-pca"), 1e-16))
    vr = np.log10(np.maximum(load_errors(folder, "vr-pca"), 1e-16))
    epochs = np.arange(1, len(pvr) + 1)

    figure, axis = plt.subplots(figsize=(5.2, 3.9))
    marker_step = max(1, len(epochs) // 10)
    axis.plot(
        epochs,
        vr,
        color="#d95f02",
        linestyle="--",
        linewidth=1.8,
        marker="s",
        markevery=marker_step,
        markersize=4.5,
        markerfacecolor="white",
        label="VR-PCA",
        zorder=2,
    )
    axis.plot(
        epochs,
        pvr,
        color="#1f77b4",
        linestyle="-",
        linewidth=1.8,
        marker="o",
        markevery=marker_step,
        markersize=4.2,
        label="PVR-PCA",
        zorder=3,
    )
    axis.set_xlabel("Epoch")
    axis.set_ylabel(r"$\log_{10}(\mathrm{relative\ subspace\ error})$")
    axis.set_title(title)
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    axis.margins(x=0.02)
    if show_gap:
        inset = axis.inset_axes([0.53, 0.13, 0.42, 0.30])
        inset.plot(epochs, vr - pvr, color="#444444", linewidth=1.2)
        inset.axhline(0.0, color="#999999", linewidth=0.7, linestyle=":")
        inset.set_title(r"$\log_{10}e_{VR}-\log_{10}e_{PVR}$", fontsize=7)
        inset.tick_params(labelsize=6)
        inset.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    plot_pair(
        ROOT / "output" / "fig3_no_oja_full30_matlab_order",
        ROOT / "revised" / "ccat_test.pdf",
        "Complete CCAT test, k=6",
        True,
    )
    plot_pair(
        ROOT / "output" / "diagnostic_epoch_separation" / "ccat_k6_warm2_c143",
        ROOT / "revised" / "CCAT_cut.pdf",
        "CCAT subset, k=6, Oja 2m, 143/t",
    )


if __name__ == "__main__":
    main()
