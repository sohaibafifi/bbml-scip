#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class Task:
    name: str
    cmd: list[str]
    cwd: str | None
    log_path: str | None
    env: dict[str, str]
    skip: bool = False


def _extract_summary_line(log_path: str | None) -> str | None:
    if not log_path:
        return None
    path = Path(log_path)
    if not path.is_file() or path.stat().st_size == 0:
        return None
    try:
        lines = path.read_text(errors="replace").splitlines()
    except OSError:
        return None
    for line in reversed(lines):
        line = line.strip()
        if line.startswith("TASK_SUMMARY ") or line.startswith("BUDGET_PROGRESS "):
            return line
    return None


def _load_manifest(path: Path) -> list[Task]:
    tasks: list[Task] = []
    with path.open() as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            data: dict[str, Any] = json.loads(line)
            cmd = data.get("cmd")
            if not isinstance(cmd, list) or not all(isinstance(part, str) for part in cmd):
                raise ValueError(f"{path}:{lineno}: expected cmd to be a string list")
            env = data.get("env") or {}
            if not isinstance(env, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in env.items()):
                raise ValueError(f"{path}:{lineno}: expected env to be a string map")
            name = data.get("name") or f"task-{lineno}"
            tasks.append(
                Task(
                    name=str(name),
                    cmd=cmd,
                    cwd=data.get("cwd"),
                    log_path=data.get("log_path"),
                    env=env,
                    skip=bool(data.get("skip", False)),
                )
            )
    return tasks


def _run_task(task: Task) -> tuple[str, int, str | None]:
    if task.skip:
        return task.name, 0, task.log_path
    env = os.environ.copy()
    env.update(task.env)
    log_path = Path(task.log_path) if task.log_path else None
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w") as log:
            proc = subprocess.run(
                task.cmd,
                cwd=task.cwd,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
    else:
        proc = subprocess.run(task.cmd, cwd=task.cwd, env=env, check=False)
    return task.name, int(proc.returncode), task.log_path


def main() -> int:
    ap = argparse.ArgumentParser(description="Run manifest-defined subprocess tasks with bounded parallelism.")
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--jobs", type=int, default=1)
    args = ap.parse_args()

    tasks = _load_manifest(args.manifest)
    total = len(tasks)
    skipped = sum(1 for task in tasks if task.skip)
    runnable = [task for task in tasks if not task.skip]

    if total == 0:
        print(f"[task-runner] no tasks in {args.manifest}")
        return 0

    failures: list[tuple[str, int, str | None]] = []
    successes = 0
    print(
        f"[task-runner] manifest={args.manifest} total={total} " f"runnable={len(runnable)} skipped={skipped} jobs={max(1, args.jobs)}",
        flush=True,
    )

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        future_map = {pool.submit(_run_task, task): task for task in runnable}
        for future in concurrent.futures.as_completed(future_map):
            task = future_map[future]
            name, returncode, log_path = future.result()
            if returncode == 0:
                successes += 1
                summary = _extract_summary_line(log_path)
                suffix = f" {summary}" if summary else ""
                print(f"[ok] {name}{suffix}", flush=True)
            else:
                failures.append((name, returncode, log_path))
                where = f" log={log_path}" if log_path else ""
                print(f"[fail] {name} exit={returncode}{where}", file=sys.stderr, flush=True)

    print(
        f"[task-runner] done success={successes} skipped={skipped} failed={len(failures)}",
        flush=True,
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
