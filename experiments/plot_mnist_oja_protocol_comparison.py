from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output" / "mnist_oja_protocol_comparison"
PROTOCOLS = {
    "Independent Oja streams": ROOT / "output" / "mnist_historical_independent_oja",
    "Shared post-Oja subspace": ROOT / "output" / "mnist_shared_oja_step3_paper",
}


def load_protocol(folder: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pvr_errors = []
    vr_errors = []
    pvr_seconds = []
    vr_seconds = []
    for run in sorted(folder.glob("k17_seed*")):
        pvr = json.loads((run / "ccat_original_pvr-pca.json").read_text())
        vr = json.loads((run / "ccat_original_vr-pca.json").read_text())
        pvr_errors.append(pvr["errors"])
        vr_errors.append(vr["errors"])
        pvr_seconds.append(pvr["epoch_seconds"])
        vr_seconds.append(vr["epoch_seconds"])
    return tuple(np.asarray(values, dtype=float) for values in (
        pvr_errors, vr_errors, pvr_seconds, vr_seconds
    ))


def main() -> None:
    figure, axes = plt.subplots(2, 2, figsize=(9.2, 7.0))
    summary = {}
    colors = {"PVR-PCA": "#1f77b4", "VR-PCA": "#d95f02"}

    for column, (name, folder) in enumerate(PROTOCOLS.items()):
        pvr_error, vr_error, pvr_time, vr_time = load_protocol(folder)
        epochs = np.arange(1, pvr_error.shape[1] + 1)
        for label, errors in (("PVR-PCA", pvr_error), ("VR-PCA", vr_error)):
            logged = np.log10(np.maximum(errors, np.finfo(float).tiny))
            mean = logged.mean(axis=0)
            standard_deviation = logged.std(axis=0, ddof=1)
            axes[0, column].plot(epochs, mean, marker="o", markersize=3.2,
                                 linewidth=1.7, color=colors[label], label=label)
            axes[0, column].fill_between(
                epochs, mean - standard_deviation, mean + standard_deviation,
                color=colors[label], alpha=0.16
            )
        axes[0, column].set_title(f"{name} (n={pvr_error.shape[0]})")
        axes[0, column].set_xlabel("Epoch")
        axes[0, column].set_ylabel(r"Mean $\log_{10}$(relative error)")
        axes[0, column].grid(alpha=0.25)
        axes[0, column].legend(frameon=False)

        final_ratio = pvr_error[:, -1] / vr_error[:, -1]
        speedup = vr_time.sum(axis=1) / pvr_time.sum(axis=1)
        positions = np.arange(len(final_ratio)) + 1
        axes[1, column].scatter(positions, final_ratio, color="#4c78a8", s=26,
                                label="Final error ratio")
        axes[1, column].axhline(1, color="black", linestyle="--", linewidth=1)
        axes[1, column].set_yscale("log")
        axes[1, column].set_xlabel("Independent run")
        axes[1, column].set_ylabel("Final error ratio PVR/VR (log scale)")
        axes[1, column].grid(alpha=0.25)
        axes[1, column].set_title(
            f"PVR wins {np.sum(final_ratio < 1)}/{len(final_ratio)}; "
            f"speedup {speedup.mean():.3f}x"
        )
        summary[name] = {
            "runs": int(len(final_ratio)),
            "pvr_final_error_wins": int(np.sum(final_ratio < 1)),
            "final_error_ratio_median": float(np.median(final_ratio)),
            "final_error_ratio_geometric_mean": float(np.exp(np.mean(np.log(final_ratio)))),
            "speedup_mean": float(speedup.mean()),
            "speedup_sample_sd": float(speedup.std(ddof=1)),
        }

    figure.suptitle("MNIST k=17: effect of the Oja warm-start protocol", y=1.01)
    figure.tight_layout()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT / "MNIST_k17_Oja_protocol_comparison.pdf", bbox_inches="tight")
    figure.savefig(OUTPUT / "MNIST_k17_Oja_protocol_comparison.png", dpi=220,
                   bbox_inches="tight")
    plt.close(figure)
    (OUTPUT / "MNIST_k17_Oja_protocol_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
