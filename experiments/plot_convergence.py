from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.linalg import eigsh

from run_benchmarks import load_data


VARIANTS = ["VR-PCA", "Projection-only", "QR-only", "PVR-PCA"]
COLORS = {"VR-PCA": "#333333", "Projection-only": "#377eb8", "QR-only": "#4daf4a", "PVR-PCA": "#e41a1c"}


def reference_energies(data, ranks: list[int]) -> dict[int, float]:
    if not hasattr(data, "multiply"):
        eigenvalues = np.linalg.eigvalsh(data.T @ data / data.shape[0])[::-1]
        return {rank: float(eigenvalues[:rank].sum()) for rank in ranks}
    eigenvalues = eigsh(data.T @ data / data.shape[0], k=max(ranks),
                        return_eigenvectors=False, which="LA")
    eigenvalues = np.sort(eigenvalues)[::-1]
    return {rank: float(eigenvalues[:rank].sum()) for rank in ranks}


def plot_dataset(dataset: str, source: Path, ranks: list[int], output: Path,
                 samples: int, seed: int) -> None:
    curves = json.loads((output / f"{dataset}_curves.json").read_text())
    data = load_data(dataset, source, samples, seed)
    energies = reference_energies(data, ranks)
    for rank in ranks:
        figure, axis = plt.subplots(figsize=(5.2, 3.8))
        for variant in VARIANTS:
            runs = []
            for key, objectives in curves.items():
                key_rank, key_variant, _ = key.split(":", 2)
                if int(key_rank) == rank and key_variant == variant:
                    errors = np.maximum(1 - np.asarray(objectives) / energies[rank], 1e-16)
                    runs.append(np.log10(errors))
            if not runs:
                continue
            values = np.vstack(runs)
            median = np.median(values, axis=0)
            epochs = np.arange(median.size)
            axis.plot(epochs, median, label=variant, color=COLORS[variant], linewidth=1.7)
            if values.shape[0] > 1:
                axis.fill_between(epochs, np.quantile(values, .25, axis=0),
                                  np.quantile(values, .75, axis=0),
                                  color=COLORS[variant], alpha=.12)
        axis.set_xlabel("Epoch")
        axis.set_ylabel(r"$\log_{10}$ relative subspace objective error")
        axis.set_title(f"{dataset.upper()}, k={rank}")
        axis.grid(alpha=.25)
        axis.legend(fontsize=8)
        figure.tight_layout()
        figure.savefig(output / f"{dataset}_convergence_k{rank}.pdf")
        plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["mnist", "ccat"], required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--ranks", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260724)
    args = parser.parse_args()
    plot_dataset(args.dataset, args.source, args.ranks, args.output,
                 args.samples, args.seed)


if __name__ == "__main__":
    main()
