from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path("output/subset_shamir_protocol")
NO_WARM = ROOT / "four_method_grid" / "k6"
WARM = ROOT / "oja_warm_ablation" / "k6_warm2"
OUTPUT = ROOT / "oja_warm_ablation"


def load(directory: Path, method: str) -> dict:
    return json.loads((directory / f"ccat_original_{method.lower()}.json").read_text())


def main() -> None:
    warm_cost = load(WARM, "PVR-PCA")["initialization_seconds"]
    rows = []
    figure, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
    styles = {
        ("PVR-PCA", "No Oja"): ("#1f77b4", "-", "o"),
        ("VR-PCA", "No Oja"): ("#d95f02", "--", "D"),
        ("PVR-PCA", "2m Oja"): ("#2ca02c", "-", "s"),
        ("VR-PCA", "2m Oja"): ("#9467bd", "--", "^"),
    }
    for method in ["PVR-PCA", "VR-PCA"]:
        for label, directory, initialization in [
            ("No Oja", NO_WARM, 0.0),
            ("2m Oja", WARM, warm_cost),
        ]:
            state = load(directory, method)
            epochs = np.arange(1, len(state["errors"]) + 1)
            elapsed = initialization + np.cumsum(state["epoch_seconds"])
            color, line_style, marker = styles[(method, label)]
            name = f"{method}, {label}"
            axes[0].semilogy(epochs, state["errors"], label=name, color=color,
                            linestyle=line_style, marker=marker, markersize=3)
            axes[1].semilogy(elapsed, state["errors"], label=name, color=color,
                            linestyle=line_style, marker=marker, markersize=3)
            rows.append({"method": method, "warm_start": label,
                         "warm_seconds": initialization,
                         "main_seconds": state["main_seconds"],
                         "end_to_end_seconds": initialization + state["main_seconds"],
                         "final_error": state["errors"][-1]})
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Relative subspace error")
    axes[0].set_title("Convergence per epoch")
    axes[1].set_xlabel("End-to-end wall-clock time (s)")
    axes[1].set_ylabel("Relative subspace error")
    axes[1].set_title("Convergence including warm start")
    for axis in axes:
        axis.grid(alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    figure.tight_layout(rect=(0, 0, 1, 0.82))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT / "CCAT_Oja_warm_start_ablation.pdf")
    figure.savefig(OUTPUT / "CCAT_Oja_warm_start_ablation.png", dpi=220)
    pd.DataFrame(rows).to_csv(OUTPUT / "CCAT_Oja_warm_start_ablation.csv", index=False)


if __name__ == "__main__":
    main()
