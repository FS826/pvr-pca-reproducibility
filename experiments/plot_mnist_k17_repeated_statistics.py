from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "output" / "mnist_k17_c243_validation"
OUTPUT_ROOT = ROOT / "output" / "mnist_k17_c243_statistics"


def load_runs(method: str) -> tuple[np.ndarray, np.ndarray]:
    errors = []
    epoch_seconds = []
    filename = f"ccat_original_{method}.json"
    for folder in sorted(RESULT_ROOT.glob("k17_seed*")):
        result = json.loads((folder / filename).read_text())
        errors.append(result["errors"])
        epoch_seconds.append(result["epoch_seconds"])
    return np.asarray(errors, dtype=float), np.asarray(epoch_seconds, dtype=float)


def main() -> None:
    pvr_errors, pvr_seconds = load_runs("pvr-pca")
    vr_errors, vr_seconds = load_runs("vr-pca")
    if pvr_errors.shape != vr_errors.shape or pvr_errors.shape[0] != 10:
        raise RuntimeError("Expected ten paired PVR-PCA and VR-PCA runs")

    epochs = np.arange(1, pvr_errors.shape[1] + 1)
    pvr_log = np.log10(np.maximum(pvr_errors, np.finfo(float).tiny))
    vr_log = np.log10(np.maximum(vr_errors, np.finfo(float).tiny))

    figure, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
    for values, color, label in (
        (vr_log, "#d95f02", "VR-PCA"),
        (pvr_log, "#1f77b4", "PVR-PCA"),
    ):
        mean = values.mean(axis=0)
        standard_deviation = values.std(axis=0, ddof=1)
        axes[0].plot(epochs, mean, linewidth=1.8, marker="o", markersize=3.5,
                     color=color, label=label)
        axes[0].fill_between(epochs, mean - standard_deviation,
                             mean + standard_deviation, color=color, alpha=0.16)

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel(r"$\log_{10}(\mathrm{relative\ subspace\ error})$")
    axes[0].set_title("MNIST, k=17 (10 independent runs)")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False)

    timing = [pvr_seconds.mean(axis=1), vr_seconds.mean(axis=1)]
    box = axes[1].boxplot(timing, tick_labels=["PVR-PCA", "VR-PCA"],
                          patch_artist=True, widths=0.55)
    for patch, color in zip(box["boxes"], ("#1f77b4", "#d95f02")):
        patch.set_facecolor(color)
        patch.set_alpha(0.45)
    axes[1].set_ylabel("Mean time per epoch (s)")
    axes[1].set_title("Wall-clock time")
    axes[1].grid(axis="y", alpha=0.25)

    figure.tight_layout()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_ROOT / "MNIST_k17_repeated_statistics.pdf",
                   bbox_inches="tight")
    figure.savefig(OUTPUT_ROOT / "MNIST_k17_repeated_statistics.png", dpi=220,
                   bbox_inches="tight")
    plt.close(figure)

    final_ratio = pvr_errors[:, -1] / vr_errors[:, -1]
    speedup = vr_seconds.sum(axis=1) / pvr_seconds.sum(axis=1)
    summary = {
        "runs": int(pvr_errors.shape[0]),
        "rank": 17,
        "warm_passes": 2,
        "warm_coefficient": 243,
        "final_error_ratio_mean": float(final_ratio.mean()),
        "final_error_ratio_sample_sd": float(final_ratio.std(ddof=1)),
        "pvr_lower_final_error_runs": int(np.sum(final_ratio < 1)),
        "speedup_mean": float(speedup.mean()),
        "speedup_sample_sd": float(speedup.std(ddof=1)),
        "speedup_min": float(speedup.min()),
        "speedup_max": float(speedup.max()),
    }
    (OUTPUT_ROOT / "MNIST_k17_repeated_statistics.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
