#!/usr/bin/env python3
"""Run a command while sampling GPU memory with nvidia-smi.

The monitor records total memory used on a physical GPU and reports peak delta
over the pre-run baseline. This is intended for experiment-level comparisons
when the GPU may already have stable resident allocations.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import subprocess
import threading
import time
from pathlib import Path
from statistics import mean


def _query_used_mib(gpu_index: int) -> int:
    out = subprocess.check_output(
        [
            "nvidia-smi",
            f"--id={gpu_index}",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    return int(out.strip().splitlines()[0].strip())


def _utc_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True, help="Physical GPU index to sample.")
    parser.add_argument("--interval", type=float, default=0.2, help="Sampling interval in seconds.")
    parser.add_argument("--csv", type=Path, required=True, help="Output sampling CSV path.")
    parser.add_argument("--summary", type=Path, required=True, help="Output summary JSON path.")
    parser.add_argument("--log", type=Path, required=True, help="Command stdout/stderr log path.")
    parser.add_argument("--", dest="dashdash", action="store_true")
    args, cmd = parser.parse_known_args()

    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        raise SystemExit("No command provided after --")

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.log.parent.mkdir(parents=True, exist_ok=True)

    baseline_samples = []
    for _ in range(5):
        baseline_samples.append(_query_used_mib(args.gpu))
        time.sleep(max(args.interval, 0.05))
    baseline_mib = int(round(mean(baseline_samples)))

    stop = threading.Event()
    samples: list[dict[str, object]] = []
    start = time.time()

    def sampler() -> None:
        while not stop.is_set():
            now = time.time()
            try:
                used = _query_used_mib(args.gpu)
            except Exception as exc:  # pragma: no cover - diagnostic path
                samples.append(
                    {
                        "time_utc": _utc_iso(),
                        "elapsed_s": now - start,
                        "gpu_index": args.gpu,
                        "used_mib": "",
                        "delta_mib": "",
                        "error": str(exc),
                    }
                )
            else:
                samples.append(
                    {
                        "time_utc": _utc_iso(),
                        "elapsed_s": now - start,
                        "gpu_index": args.gpu,
                        "used_mib": used,
                        "delta_mib": used - baseline_mib,
                        "error": "",
                    }
                )
            stop.wait(args.interval)

    thread = threading.Thread(target=sampler, daemon=True)
    thread.start()

    with args.log.open("w", encoding="utf-8") as log_f:
        log_f.write(f"START_TIME={dt.datetime.now().astimezone().isoformat()}\n")
        log_f.write(f"GPU_BASELINE_MIB={baseline_mib}\n")
        log_f.write("COMMAND=" + " ".join(cmd) + "\n")
        log_f.flush()
        proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT)
        return_code = proc.wait()
        log_f.write(f"EXIT_STATUS={return_code}\n")
        log_f.write(f"END_TIME={dt.datetime.now().astimezone().isoformat()}\n")

    stop.set()
    thread.join(timeout=max(args.interval * 2, 1.0))

    with args.csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["time_utc", "elapsed_s", "gpu_index", "used_mib", "delta_mib", "error"],
        )
        writer.writeheader()
        writer.writerows(samples)

    numeric_used = [int(s["used_mib"]) for s in samples if s.get("used_mib") != ""]
    numeric_delta = [int(s["delta_mib"]) for s in samples if s.get("delta_mib") != ""]
    summary = {
        "gpu_index": args.gpu,
        "command": cmd,
        "return_code": return_code,
        "baseline_mib": baseline_mib,
        "baseline_samples_mib": baseline_samples,
        "sample_interval_s": args.interval,
        "num_samples": len(samples),
        "peak_used_mib": max(numeric_used) if numeric_used else None,
        "peak_delta_mib": max(numeric_delta) if numeric_delta else None,
        "mean_delta_mib": mean(numeric_delta) if numeric_delta else None,
        "min_delta_mib": min(numeric_delta) if numeric_delta else None,
    }
    args.summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
