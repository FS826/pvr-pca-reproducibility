from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "output" / "fig3_no_oja_full30_matlab_order"
OUTPUT = ROOT / "output" / "ccat_full_time_accuracy"


def load(method: str) -> tuple[np.ndarray, np.ndarray]:
    path = RESULTS / f"ccat_original_{method}.json"
    record = json.loads(path.read_text())
    elapsed_minutes = np.cumsum(record["epoch_seconds"]) / 60.0
    errors = np.asarray(record["errors"], dtype=float)
    return elapsed_minutes, errors


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(5.8, 4.0))
    styles = {
        "pvr-pca": ("PVR-PCA", "#d62728", "o"),
        "vr-pca": ("VR-PCA", "#1f77b4", "s"),
    }
    for method, (label, color, marker) in styles.items():
        minutes, errors = load(method)
        axis.semilogy(
            minutes,
            errors,
            label=label,
            color=color,
            marker=marker,
            markevery=3,
            markersize=4,
            linewidth=1.8,
        )
    axis.set_xlabel("Cumulative wall-clock time (minutes)")
    axis.set_ylabel("Relative subspace error")
    axis.grid(True, which="both", linestyle=":", linewidth=0.7, alpha=0.7)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(OUTPUT / "CCAT_full_time_accuracy.pdf", bbox_inches="tight")
    figure.savefig(OUTPUT / "CCAT_full_time_accuracy.png", dpi=240, bbox_inches="tight")


if __name__ == "__main__":
    main()
