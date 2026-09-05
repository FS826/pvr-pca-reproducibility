from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    result_root = root / "output" / "subset_shamir_protocol"
    summary = pd.read_csv(result_root / "ccat_paper_80000x880_k_sweep_summary.csv")
    output = result_root / "CCAT_80000x880_k_sweep.pdf"

    figure = plt.figure(figsize=(11.2, 7.4))
    grid = figure.add_gridspec(2, 3, height_ratios=[1.25, 1], hspace=0.42, wspace=0.30)
    colors = {"PVR-PCA": "#d62728", "VR-PCA": "#1f77b4"}

    for column, rank in enumerate((6, 16, 24)):
        axis = figure.add_subplot(grid[0, column])
        folder = result_root / f"ccat_paper_80000x880_k{rank}"
        for method in ("PVR-PCA", "VR-PCA"):
            path = folder / f"ccat_original_{method.lower()}.json"
            result = json.loads(path.read_text())
            epochs = np.arange(1, len(result["errors"]) + 1)
            axis.plot(epochs, np.log10(np.maximum(result["errors"], 1e-16)),
                      "-o", label=method, color=colors[method], markersize=3.2)
        axis.set_title(f"CCAT subset, k={rank}")
        axis.set_xlabel("Epoch")
        axis.set_ylabel(r"$\log_{10}$ relative objective error")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)

    speed_axis = figure.add_subplot(grid[1, :2])
    speed_axis.plot(summary["k"], summary["speedup"], "-o", color="#2ca02c", linewidth=2)
    speed_axis.axhline(1, color="#555555", linestyle="--", linewidth=1)
    speed_axis.set_xlabel("Target rank k")
    speed_axis.set_ylabel("VR-PCA time / PVR-PCA time")
    speed_axis.set_title("Per-10-epoch wall-clock speedup")
    speed_axis.set_xticks(summary["k"])
    speed_axis.grid(alpha=0.25)

    ratio_axis = figure.add_subplot(grid[1, 2])
    ratios = summary["pvr_error_ratio_vs_vr"]
    bars = ratio_axis.bar(summary["k"].astype(str), ratios,
                          color=np.where(ratios <= 1, "#d62728", "#999999"))
    ratio_axis.axhline(1, color="#555555", linestyle="--", linewidth=1)
    ratio_axis.set_xlabel("Target rank k")
    ratio_axis.set_ylabel("PVR final error / VR final error")
    ratio_axis.set_title("Final-error ratio")
    ratio_axis.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, ratios):
        ratio_axis.text(bar.get_x() + bar.get_width() / 2, value + 0.08,
                        f"{value:.2f}", ha="center", va="bottom", fontsize=7)

    figure.suptitle("CCAT paper-size subset: n=80000, d=880", fontsize=14)
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)
    print(output)


if __name__ == "__main__":
    main()
