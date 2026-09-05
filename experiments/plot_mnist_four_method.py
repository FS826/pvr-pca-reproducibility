from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path("output/mnist_four_method_grid")
METHODS = ["PVR-PCA", "Projection-only", "QR-only", "VR-PCA"]
STYLES = {
    "PVR-PCA": ("#1f77b4", "-", "o"),
    "Projection-only": ("#2ca02c", "-.", "s"),
    "QR-only": ("#9467bd", ":", "^"),
    "VR-PCA": ("#d95f02", "--", "D"),
}


def load_state(rank: int, method: str) -> dict:
    path = ROOT / f"k{rank}" / f"ccat_original_{method.lower()}.json"
    return json.loads(path.read_text())


def main() -> None:
    rows = []
    figure, axes = plt.subplots(1, 3, figsize=(13.2, 3.8))
    for axis, rank in zip(axes[:2], [17, 28]):
        for method in METHODS:
            state = load_state(rank, method)
            elapsed = np.cumsum(state["epoch_seconds"])
            color, line_style, marker = STYLES[method]
            axis.semilogy(elapsed, state["errors"], label=method, color=color,
                          linestyle=line_style, marker=marker, markersize=3,
                          linewidth=1.6)
            rows.append({"rank": rank, "method": method,
                         "epochs": len(state["errors"]),
                         "final_error": state["errors"][-1],
                         "main_seconds": state["main_seconds"],
                         "seconds_per_epoch": np.mean(state["epoch_seconds"]),
                         "step_coefficient": state["step_coefficient"]})
        axis.set_title(f"MNIST, k={rank}")
        axis.set_xlabel("Cumulative wall-clock time (s)")
        axis.set_ylabel("Relative subspace error")
        axis.grid(alpha=0.25)

    benchmark = pd.read_csv("output/python_formal/mnist_summary.csv")
    for method in METHODS:
        group = benchmark[benchmark["variant"] == method].sort_values("rank")
        color, line_style, marker = STYLES[method]
        axes[2].plot(group["rank"], group["seconds_median"], label=method,
                     color=color, linestyle=line_style, marker=marker,
                     linewidth=1.6)
    axes[2].set_title("Controlled 3,000-update runtime")
    axes[2].set_xlabel("Target rank k")
    axes[2].set_ylabel("Median wall-clock time (s)")
    axes[2].grid(alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    figure.tight_layout(rect=(0, 0, 1, 0.89))
    ROOT.mkdir(parents=True, exist_ok=True)
    figure.savefig(ROOT / "MNIST_four_method_comparison.pdf")
    figure.savefig(ROOT / "MNIST_four_method_comparison.png", dpi=220)
    pd.DataFrame(rows).to_csv(ROOT / "MNIST_four_method_summary.csv", index=False)


if __name__ == "__main__":
    main()
