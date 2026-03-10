import argparse
import time
from typing import List

import numpy as np
import onnxruntime as ort
from bbml.train.train_rank import DEFAULT_FEATS


def _ort_numpy_dtype(type_str: str) -> np.dtype:
    if "float16" in type_str:
        return np.float16
    if "double" in type_str or "float64" in type_str:
        return np.float64
    return np.float32


def _build_inputs(session: ort.InferenceSession, n_var: int, d_var: int, d_con: int, edge_factor: int) -> dict[str, np.ndarray]:
    inputs = session.get_inputs()
    x_dtype = _ort_numpy_dtype(inputs[0].type)
    if len(inputs) <= 1:
        return {inputs[0].name: np.random.randn(n_var, d_var).astype(x_dtype)}

    n_con = max(1, n_var // 2)
    n_edge = max(1, n_var * edge_factor)
    rows = np.random.randint(0, n_con, size=(n_edge,), dtype=np.int64)
    cols = np.random.randint(0, n_var, size=(n_edge,), dtype=np.int64)
    c_dtype = _ort_numpy_dtype(inputs[1].type)
    return {
        inputs[0].name: np.random.randn(n_var, d_var).astype(x_dtype),
        inputs[1].name: np.random.randn(n_con, d_con).astype(c_dtype),
        inputs[2].name: np.stack([rows, cols], axis=0),
    }


def _preferred_providers() -> List[str]:
    preferred = [
        "CoreMLExecutionProvider",
        "CUDAExecutionProvider",
        "DmlExecutionProvider",
        "CPUExecutionProvider",
    ]
    available = ort.get_available_providers()
    ordered = [provider for provider in preferred if provider in available]
    return ordered or available


def bench(onnx_path: str, dims: List[int], d_var: int, d_con: int, runs: int = 50, edge_factor: int = 2):
    sess = ort.InferenceSession(onnx_path, providers=_preferred_providers())
    print(f"providers={sess.get_providers()}")
    for n_var in dims:
        feed = _build_inputs(sess, n_var=n_var, d_var=d_var, d_con=d_con, edge_factor=edge_factor)
        try:
            for _ in range(5):
                sess.run(None, feed)
            t0 = time.perf_counter()
            for _ in range(runs):
                sess.run(None, feed)
            dt = (time.perf_counter() - t0) / runs
            print(f"n_var={n_var:4d}  avg={dt*1e3:7.3f} ms")
        except Exception as exc:
            print(f"unsupported on current provider(s): {exc}")
            return


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
