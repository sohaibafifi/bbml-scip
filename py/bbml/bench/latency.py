import argparse
import time
from typing import List

import numpy as np
import onnxruntime as ort


def bench(onnx_path: str, dims: List[int], d: int, runs: int = 50):
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    for m in dims:
        X = np.random.randn(m, d).astype(np.float32)
        # Warmup
        for _ in range(5):
            sess.run(None, {name: X})
        t0 = time.perf_counter()
        for _ in range(runs):
            sess.run(None, {name: X})
        dt = (time.perf_counter() - t0) / runs
        print(f"m={m:4d}  avg={dt*1e3:7.3f} ms")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True, help="Path to ONNX model")
    ap.add_argument("--d", type=int, default=6, help="Input feature dimension")
    ap.add_argument(
        "--dims",
        type=int,
        nargs="*",
        default=[100, 250, 500, 1000, 1500, 2000],
        help="Candidate counts to benchmark",
    )
    ap.add_argument("--runs", type=int, default=50)
    args = ap.parse_args()
    bench(args.onnx, args.dims, args.d, runs=args.runs)


if __name__ == "__main__":
    main()
