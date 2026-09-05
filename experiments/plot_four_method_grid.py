from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path("output/subset_shamir_protocol")
OUTPUT = ROOT / "four_method_grid"
METHODS = ["PVR-PCA", "Projection-only", "QR-only", "VR-PCA"]
STYLES = {
    "PVR-PCA": ("#1f77b4", "-", "o"),
    "Projection-only": ("#2ca02c", "-.", "s"),
    "QR-only": ("#9467bd", ":", "^"),
    "VR-PCA": ("#d95f02", "--", "D"),
}


def load_state(directory: Path, method: str) -> dict:
    return json.loads((directory / f"ccat_original_{method.lower()}.json").read_text())


def main() -> None:
    rows = []
    figure, axes = plt.subplots(1, 3, figsize=(13.2, 3.8))
    for axis, rank in zip(axes[:2], [6, 24]):
        directory = OUTPUT / f"k{rank}"
        for method in METHODS:
            state = load_state(directory, method)
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
                         "step_coefficient": (state["step_coefficient"]
                                              if state["step_coefficient"] is not None
                                              else 1 / rank)})
        axis.set_title(f"CCAT subset, k={rank}")
        axis.set_xlabel("Cumulative wall-clock time (s)")
        axis.set_ylabel("Relative subspace error")
        axis.grid(alpha=0.25)

    for method in METHODS:
        state = load_state(OUTPUT / "k16", method)
        rows.append({"rank": 16, "method": method,
                     "epochs": len(state["errors"]),
                     "final_error": state["errors"][-1],
                     "main_seconds": state["main_seconds"],
                     "seconds_per_epoch": np.mean(state["epoch_seconds"]),
                     "step_coefficient": (state["step_coefficient"]
                                          if state["step_coefficient"] is not None
                                          else 1 / 16)})

    large_directories = {
        48: ROOT / "pilot_k48_scaled",
        64: ROOT / "pilot_k64_four_c1over64",
        96: ROOT / "pilot_k96_four_c1over96",
    }
    for method in METHODS:
        times = []
        for rank, directory in large_directories.items():
            state = load_state(directory, method)
            times.append(state["epoch_seconds"][-1])
            rows.append({"rank": rank, "method": method,
                         "epochs": len(state["errors"]),
                         "final_error": state["errors"][-1],
                         "main_seconds": state["main_seconds"],
                         "seconds_per_epoch": np.mean(state["epoch_seconds"]),
                         "step_coefficient": (state["step_coefficient"]
                                              if state["step_coefficient"] is not None
                                              else 1 / rank)})
        color, line_style, marker = STYLES[method]
        axes[2].plot(list(large_directories), times, label=method, color=color,
                     linestyle=line_style, marker=marker, linewidth=1.6)
    axes[2].set_title("Large-k one-epoch runtime")
    axes[2].set_xlabel("Target rank k")
    axes[2].set_ylabel("Wall-clock time (s)")
    axes[2].grid(alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    figure.tight_layout(rect=(0, 0, 1, 0.89))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT / "CCAT_four_method_comparison.pdf")
    figure.savefig(OUTPUT / "CCAT_four_method_comparison.png", dpi=220)
    pd.DataFrame(rows).to_csv(OUTPUT / "CCAT_four_method_summary.csv", index=False)


if __name__ == "__main__":
    main()
