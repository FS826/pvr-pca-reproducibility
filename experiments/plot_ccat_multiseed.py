from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def result_folder(root: Path, seed: int, rank: int) -> Path:
    if seed == 20260808:
        return root / f"ccat_paper_80000x880_k{rank}"
    return root / "multiseed" / f"seed{seed}" / f"k{rank}"


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    root = project / "output" / "subset_shamir_protocol"
    summary = pd.read_csv(root / "ccat_10seed_summary.csv")
    seeds = tuple(range(20260808, 20260818))
    ranks = (6, 16, 24)
    colors = {"PVR-PCA": "#d62728", "VR-PCA": "#1f77b4"}

    figure = plt.figure(figsize=(11.2, 7.3))
    grid = figure.add_gridspec(2, 3, height_ratios=[1.3, 1], hspace=0.42, wspace=0.30)
    for column, rank in enumerate(ranks):
        axis = figure.add_subplot(grid[0, column])
        for method in ("PVR-PCA", "VR-PCA"):
            runs = []
            time_runs = []
            for seed in seeds:
                folder = result_folder(root, seed, rank)
                result = json.loads((folder / f"ccat_original_{method.lower()}.json").read_text())
                runs.append(np.log10(np.maximum(result["errors"], 1e-16)))
                time_runs.append(np.cumsum(result["epoch_seconds"]))
            values = np.asarray(runs)
            times = np.asarray(time_runs)
            mean_times = times.mean(axis=0)
            mean = values.mean(axis=0)
            std = values.std(axis=0, ddof=1)
            axis.plot(mean_times, mean, "-o", color=colors[method], label=method, markersize=3)
            axis.fill_between(mean_times, mean - std, mean + std,
                              color=colors[method], alpha=0.14)
        axis.set_title(f"k={rank}")
        axis.set_xlabel("Mean cumulative wall-clock time (s)")
        axis.set_ylabel(r"Mean $\log_{10}$ relative error")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)

    speed_axis = figure.add_subplot(grid[1, :2])
    speed_axis.errorbar(summary["k"], summary["speedup_mean"],
                        yerr=summary["speedup_std"], fmt="-o", capsize=4,
                        color="#2ca02c", linewidth=2)
    speed_axis.axhline(1, color="#555555", linestyle="--", linewidth=1)
    speed_axis.set_xticks(summary["k"])
    speed_axis.set_xlabel("Target rank k")
    speed_axis.set_ylabel("VR-PCA time / PVR-PCA time")
    speed_axis.set_title("Mean wall-clock speedup (10 seeds)")
    speed_axis.grid(alpha=0.25)

    win_axis = figure.add_subplot(grid[1, 2])
    bars = win_axis.bar(summary["k"].astype(str), summary["accuracy_wins"], color="#d62728")
    win_axis.set_ylim(0, 10.8)
    win_axis.set_yticks((0, 2, 4, 6, 8, 10))
    win_axis.set_xlabel("Target rank k")
    win_axis.set_ylabel("PVR-PCA accuracy wins")
    win_axis.set_title("Wins among 10 seeds")
    win_axis.grid(axis="y", alpha=0.25)
    for bar, wins in zip(bars, summary["accuracy_wins"]):
        win_axis.text(bar.get_x() + bar.get_width() / 2, wins + 0.08,
                      f"{int(wins)}/10", ha="center", fontsize=9)

    figure.suptitle("CCAT subset (n=80000, d=880): ten-seed comparison", fontsize=14)
    output = root / "CCAT_80000x880_10seed_time.pdf"
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)
    print(output)


if __name__ == "__main__":
    main()
