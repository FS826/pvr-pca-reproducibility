from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "output" / "diagnostic_epoch_separation"


def errors(folder: str, method: str) -> np.ndarray:
    path = RESULTS / folder / f"ccat_original_{method.lower()}.json"
    return np.asarray(json.loads(path.read_text())["errors"], dtype=float)


def draw(axis, folder: str, title: str) -> None:
    pvr = np.log10(np.maximum(errors(folder, "pvr-pca"), 1e-16))
    vr = np.log10(np.maximum(errors(folder, "vr-pca"), 1e-16))
    epochs = np.arange(1, len(pvr) + 1)
    axis.plot(epochs, vr, "--s", color="#d95f02", markerfacecolor="white",
              linewidth=1.5, markersize=3.8, label="VR-PCA")
    axis.plot(epochs, pvr, "-o", color="#1f77b4", linewidth=1.5,
              markersize=3.6, label="PVR-PCA")
    axis.set_title(title)
    axis.set_xlabel("Epoch")
    axis.set_ylabel(r"$\log_{10}(e)$")
    axis.grid(alpha=0.25)


def main() -> None:
    settings = [
        ("ccat_k6", "No Oja warm-up"),
        ("ccat_k6_warm1_c81", r"Oja $m$, $81/t$"),
        ("ccat_k6_warm1_c143", r"Oja $m$, $143/t$"),
        ("ccat_k6_warm2_c81", r"Oja $2m$, $81/t$"),
        ("ccat_k6_warm2_c143", r"Oja $2m$, $143/t$"),
    ]
    figure, axes = plt.subplots(2, 3, figsize=(11.2, 6.5))
    for axis, (folder, title) in zip(axes.flat, settings):
        draw(axis, folder, title)
    axes.flat[-1].axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower right", bbox_to_anchor=(0.91, 0.12),
                  frameon=False)
    figure.suptitle("CCAT test subset, k=6: common initial subspace and independent streams")
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(RESULTS / "CCAT_k6_warm_start_sensitivity.pdf", bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
