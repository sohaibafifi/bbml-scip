import argparse
import time
from typing import List

import numpy as np
import onnxruntime as ort
from bbml.train.train_rank import DEFAULT_FEATS


def _build_inputs(session: ort.InferenceSession, n_var: int, d_var: int, d_con: int, edge_factor: int) -> dict[str, np.ndarray]:
    inputs = session.get_inputs()
    if len(inputs) <= 1:
        return {inputs[0].name: np.random.randn(n_var, d_var).astype(np.float32)}

    n_con = max(1, n_var // 2)
    n_edge = max(1, n_var * edge_factor)
    rows = np.random.randint(0, n_con, size=(n_edge,), dtype=np.int64)
    cols = np.random.randint(0, n_var, size=(n_edge,), dtype=np.int64)
    return {
        inputs[0].name: np.random.randn(n_var, d_var).astype(np.float32),
        inputs[1].name: np.random.randn(n_con, d_con).astype(np.float32),
        inputs[2].name: np.stack([rows, cols], axis=0),
    }


def bench(onnx_path: str, dims: List[int], d_var: int, d_con: int, runs: int = 50, edge_factor: int = 2):
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    for n_var in dims:
        feed = _build_inputs(sess, n_var=n_var, d_var=d_var, d_con=d_con, edge_factor=edge_factor)
        for _ in range(5):
            sess.run(None, feed)
        t0 = time.perf_counter()
        for _ in range(runs):
            sess.run(None, feed)
        dt = (time.perf_counter() - t0) / runs
        print(f"n_var={n_var:4d}  avg={dt*1e3:7.3f} ms")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True, help="Path to ONNX model")
    ap.add_argument("--d", type=int, default=len(DEFAULT_FEATS), help="Input feature dimension for single-input models")
    ap.add_argument("--d_var", type=int, default=None, help="Variable feature dimension for graph models")
    ap.add_argument("--d_con", type=int, default=None, help="Constraint feature dimension for graph models")
    ap.add_argument("--edge-factor", type=int, default=2, help="Number of graph edges per variable for graph latency smoke tests")
    ap.add_argument(
        "--dims",
        type=int,
        nargs="*",
        default=[100, 250, 500, 1000, 1500, 2000],
        help="Candidate counts to benchmark",
    )
    ap.add_argument("--runs", type=int, default=50)
    args = ap.parse_args()
    d_var = args.d_var if args.d_var is not None else args.d
    d_con = args.d_con if args.d_con is not None else d_var
    bench(args.onnx, args.dims, d_var=d_var, d_con=d_con, runs=args.runs, edge_factor=args.edge_factor)


if __name__ == "__main__":
    main()
