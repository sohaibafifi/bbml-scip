#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import inspect
import sys
import traceback
from pathlib import Path
from tempfile import TemporaryDirectory


def _load_module(path: Path, index: int):
    spec = importlib.util.spec_from_file_location(f"bbml_test_{index}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load test module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_test(fn) -> None:
    params = inspect.signature(fn).parameters
    if not params:
        fn()
        return
    if list(params) == ["tmp_path"]:
        with TemporaryDirectory() as tmp_dir:
            fn(Path(tmp_dir))
        return
    raise TypeError(f"Unsupported test fixture signature for {fn.__name__}: {list(params)}")


def main() -> int:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("tests/py")
    files = sorted(root.glob("test_*.py"))
    total = 0
    failures = 0

    for index, path in enumerate(files):
        module = _load_module(path, index)
        for name in sorted(dir(module)):
            if not name.startswith("test_"):
                continue
            fn = getattr(module, name)
            if not callable(fn):
                continue
            total += 1
            try:
                _run_test(fn)
                print(f"[ok] {path.name}::{name}")
            except Exception:
                failures += 1
                print(f"[fail] {path.name}::{name}", file=sys.stderr)
                traceback.print_exc()

    print(f"Ran {total} Python tests")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
