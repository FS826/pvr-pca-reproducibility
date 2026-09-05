from __future__ import annotations

import argparse
import json
import math
import shutil
import urllib.request
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.datasets import load_svmlight_file


DATASETS = {
    "mnist": {
        "url": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/mnist.bz2",
        "download_name": "mnist.bz2",
        "features": 780,
        "multilabel": False,
    },
    "ccat-train": {
        "url": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multilabel/rcv1_topics_train.svm.bz2",
        "download_name": "rcv1_topics_train.svm.bz2",
        "features": 47236,
        "multilabel": True,
    },
    "ccat-test": {
        "url": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multilabel/rcv1_topics_combined_test.svm.bz2",
        "download_name": "rcv1_topics_combined_test.svm.bz2",
        "features": 47236,
        "multilabel": True,
    },
}


def download(url: str, destination: Path) -> None:
    if destination.exists():
        print(f"Using existing download: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "pvr-pca-reproducibility"})
    with urllib.request.urlopen(request) as response, temporary.open("wb") as output:
        shutil.copyfileobj(response, output)
    temporary.replace(destination)
    print(f"Downloaded: {destination}")


def select_subset(matrix, rows: int, features: int):
    if rows > 0:
        matrix = matrix[:rows]
    if features > 0:
        matrix = matrix[:, :features]
    return matrix


def prepare_mnist(source: Path, output_dir: Path, rows: int, features: int) -> Path:
    matrix, _ = load_svmlight_file(source, n_features=780)
    matrix = np.asarray(matrix.toarray(), dtype=np.float64, order="C")
    matrix = select_subset(matrix, rows, features)
    matrix -= matrix.mean(axis=0, keepdims=True)
    standard_deviation = matrix.std(axis=0, ddof=1)
    nonconstant = standard_deviation > np.finfo(np.float64).eps
    matrix[:, nonconstant] /= standard_deviation[nonconstant]
    matrix[:, nonconstant] /= math.sqrt(matrix.shape[1])
    subset = "" if rows == 0 and features == 0 else f"_{matrix.shape[0]}x{matrix.shape[1]}"
    output = output_dir / f"mnist_train{subset}.npy"
    np.save(output, matrix)
    return output


def prepare_ccat(dataset: str, source: Path, output_dir: Path,
                 rows: int, features: int) -> Path:
    matrix, _ = load_svmlight_file(
        source,
        n_features=47236,
        multilabel=bool(DATASETS[dataset]["multilabel"]),
    )
    matrix = select_subset(matrix.astype(np.float64).tocsr(), rows, features)
    suffix = "training" if dataset == "ccat-train" else "test"
    subset = "" if rows == 0 and features == 0 else f"_{matrix.shape[0]}x{matrix.shape[1]}"
    output = output_dir / f"ccat_{suffix}{subset}_csr.npz"
    sparse.save_npz(output, matrix)
    return output


def metadata(path: Path, matrix) -> dict[str, object]:
    if sparse.issparse(matrix):
        squared_norm = float(matrix.multiply(matrix).sum())
        nonzeros = int(matrix.nnz)
    else:
        squared_norm = float(np.sum(matrix * matrix))
        nonzeros = int(np.count_nonzero(matrix))
    return {
        "output": str(path),
        "shape": list(matrix.shape),
        "nonzeros": nonzeros,
        "squared_frobenius_norm": squared_norm,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=DATASETS)
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--rows", type=int, default=0)
    parser.add_argument("--features", type=int, default=0)
    args = parser.parse_args()
    if args.rows < 0 or args.features < 0:
        parser.error("--rows and --features must be nonnegative")

    settings = DATASETS[args.dataset]
    download_dir = args.output_dir / "downloads"
    source = download_dir / str(settings["download_name"])
    download(str(settings["url"]), source)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset == "mnist":
        output = prepare_mnist(source, args.output_dir, args.rows, args.features)
        matrix = np.load(output, mmap_mode="r")
    else:
        output = prepare_ccat(args.dataset, source, args.output_dir,
                              args.rows, args.features)
        matrix = sparse.load_npz(output)

    report = {
        "dataset": args.dataset,
        "source_url": settings["url"],
        "source_file": str(source),
        "rows_requested": args.rows,
        "features_requested": args.features,
        **metadata(output, matrix),
    }
    report_path = output.with_suffix(output.suffix + ".json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
