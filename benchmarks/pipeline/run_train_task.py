#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description="Run a training command and write a completion marker on success.")
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--meta", required=True, type=Path)
    ap.add_argument("--signature-json", required=True)
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    args = ap.parse_args()

    cmd = list(args.cmd)
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        raise SystemExit("missing training command after --")

    signature = json.loads(args.signature_json)
    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0:
        return int(proc.returncode)

    if not args.ckpt.is_file() or args.ckpt.stat().st_size <= 0:
        print(f"run_train_task: expected non-empty checkpoint at {args.ckpt}", file=sys.stderr)
        return 2

    args.meta.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "signature": signature,
        "ckpt": str(args.ckpt),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "cmd": cmd,
    }
    tmp = args.meta.with_suffix(args.meta.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(args.meta)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
