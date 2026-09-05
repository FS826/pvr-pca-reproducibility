# PVR-PCA Reproducibility Package

This repository contains the anonymized Python implementation and experiment
drivers for the PVR-PCA study. It includes the four controlled variants used in
the ablation study:

- VR-PCA: Procrustes alignment and polar/SVD orthogonalization.
- Projection-only: least-squares alignment and polar/SVD orthogonalization.
- QR-only: Procrustes alignment and reduced Householder QR.
- PVR-PCA: least-squares alignment and reduced Householder QR.

No manuscript source, reviewer correspondence, author metadata, local paths,
credentials, or redistributed dataset files are included.

## Environment

Python 3.11 or later is recommended.

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r experiments/requirements.txt
```

For controlled wall-clock measurements, use one numerical-library thread:

```bash
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
```

Record the software and hardware environment with:

```bash
python experiments/environment_report.py --output output/environment.json
```

## Data Acquisition

The datasets are not redistributed. `experiments/download_data.py` downloads
the public LIBSVM files and converts them into the exact matrix orientation
used by the Python experiments.

Prepare the 60,000 by 780 MNIST training matrix:

```bash
python experiments/download_data.py mnist --output-dir data
```

The LIBSVM representation has 780 feature coordinates. The script centers each
feature and scales each nonconstant feature to sample standard deviation
`1/sqrt(780)` (using `ddof=1`). The 63 zero-variance coordinates are retained
as zero columns.

Prepare the sparse 23,149 by 47,236 CCAT/RCV1-topic training matrix:

```bash
python experiments/download_data.py ccat-train --output-dir data
```

Prepare the sparse 781,265 by 47,236 combined test matrix:

```bash
python experiments/download_data.py ccat-test --output-dir data
```

The CCAT matrices remain sparse and are not centered. Labels are ignored because
the experiments use only covariance products. Use `--rows` and `--features` to
create a deterministic leading-row/leading-feature subset, for example:

```bash
python experiments/download_data.py ccat-test --output-dir data \
  --rows 80000 --features 880
```

## Running Experiments

For a smoke test that requires no external dataset:

```bash
python experiments/run_synthetic.py \
  --samples 2000 --features 50 --rank 5 --epochs 2 \
  --epoch-length 100 --repeats 1 --output output/smoke
```

The main driver centralizes dataset paths, ranks, epoch budgets, repeated seeds,
and experiment switches:

```bash
python experiments/main_experiments.py
```

Edit only the configuration block at the beginning of that file. Each run
creates a timestamped directory containing results, figures, logs, and exact
environment metadata.

Representative controlled benchmarks can also be run directly:

```bash
python experiments/run_benchmarks.py \
  --dataset mnist --source data/mnist_train.npy \
  --ranks 3 6 17 28 50 68 --epochs 30 --epoch-length 60000 \
  --repeats 5 --output output/mnist_rank_sweep

python experiments/run_benchmarks.py \
  --dataset ccat --source data/ccat_training_csr.npz \
  --ranks 6 16 24 --epochs 30 --epoch-length 23149 \
  --repeats 10 --output output/ccat_rank_sweep
```

The complete CCAT-test experiment is computationally expensive:

```bash
python experiments/run_ccat_original_protocol.py \
  --data data/ccat_test_csr.npz --output output/ccat_complete \
  --rank 6 --epochs 30 --warm-passes 0 \
  --step-coefficient 0.16666666666666666 \
  --orth-backend paper --method both
```

## Reproducibility Notes

- Paired method comparisons use the same initial subspace.
- Independent random generators produce the per-method i.i.d. sample streams.
- Seeds, sampled-index records, timings, and environment metadata are written
  to the selected output directory.
- Timing excludes data download and plotting unless a script explicitly states
  otherwise.
- The full CCAT test run requires substantial memory and compute time; the
  deterministic subset command above is provided for a smaller verification.

## Dataset Sources

- MNIST LIBSVM page: <https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass.html#mnist>
- RCV1 topic data: <https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multilabel.html#rcv1v2-(topics;-full-sets)>
