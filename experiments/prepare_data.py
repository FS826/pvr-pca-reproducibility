from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.sparse import issparse, save_npz


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--variable", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--transpose", action="store_true")
    parser.add_argument("--center", action="store_true")
    parser.add_argument("--rows", type=int)
    parser.add_argument("--features", type=int)
    args = parser.parse_args()

    data = loadmat(args.source, variable_names=[args.variable])[args.variable]
    if args.transpose:
        data = data.T
    if args.rows is not None:
        data = data[:args.rows]
    if args.features is not None:
        data = data[:, :args.features]
    args.output.parent.mkdir(parents=True, exist_ok=True)

    if issparse(data):
        if args.center:
            parser.error("centering a sparse matrix would densify it")
        data = data.astype(np.float64).tocsr()
        save_npz(args.output, data)
        squared_norm = float(data.multiply(data).sum())
        nonzeros = int(data.nnz)
    else:
        data = np.asarray(data, dtype=np.float64, order="C")
        if args.center:
            data -= data.mean(axis=0, keepdims=True)
        np.save(args.output, data)
        squared_norm = float(np.sum(data * data))
        nonzeros = int(np.count_nonzero(data))

    metadata = {
        "source_variable": args.variable,
        "shape": list(data.shape),
        "nonzeros": nonzeros,
        "squared_frobenius_norm": squared_norm,
        "transposed": args.transpose,
        "centered": args.center,
        "leading_rows": args.rows,
        "leading_features": args.features,
    }
    args.output.with_suffix(args.output.suffix + ".json").write_text(
        json.dumps(metadata, indent=2)
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
