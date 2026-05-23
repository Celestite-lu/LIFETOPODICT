#!/usr/bin/env python3
"""Run a command while sampling process-tree CPU memory.

The monitor samples RSS for the launched process plus all live children. This
is intended for classifier-only feature-cache experiments where training is
CPU-bound and may use multiprocessing workers.
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

import psutil


_MIB = 1024 * 1024


def _local_iso() -> str:
    return dt.datetime.now().astimezone().isoformat()


def _utc_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _process_tree_memory(pid: int) -> tuple[float, float | None, float | None, int, str]:
    """Return process-tree RSS/PSS/USS MiB, live process count, and errors.

    RSS is useful as an upper bound, but it double counts shared pages across
    forked multiprocessing workers. PSS is the preferred Linux metric because
    shared pages are divided proportionally across processes.
    """
    try:
        root = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return 0.0, None, None, 0, ""

    processes = [root]
    try:
        processes.extend(root.children(recursive=True))
    except psutil.Error as exc:
        return 0.0, None, None, 0, str(exc)

    rss_bytes = 0
    pss_bytes = 0
    uss_bytes = 0
    have_pss = False
    have_uss = False
    live_count = 0
    errors = []
    for proc in processes:
        try:
            info = proc.memory_full_info()
            rss_bytes += info.rss
            if hasattr(info, "pss"):
                pss_bytes += info.pss
                have_pss = True
            if hasattr(info, "uss"):
                uss_bytes += info.uss
                have_uss = True
            live_count += 1
        except psutil.NoSuchProcess:
            continue
        except psutil.Error as exc:
            errors.append(f"{proc.pid}:{exc}")

    return (
        rss_bytes / _MIB,
        (pss_bytes / _MIB) if have_pss else None,
        (uss_bytes / _MIB) if have_uss else None,
        live_count,
        "; ".join(errors),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=float, default=0.1, help="Sampling interval in seconds.")
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

    stop = threading.Event()
    samples: list[dict[str, object]] = []
    start = time.time()

    with args.log.open("w", encoding="utf-8") as log_f:
        log_f.write(f"START_TIME={_local_iso()}\n")
        log_f.write("COMMAND=" + " ".join(cmd) + "\n")
        log_f.flush()
        proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT)

        def sampler() -> None:
            while not stop.is_set():
                now = time.time()
                rss_mib, pss_mib, uss_mib, process_count, error = _process_tree_memory(proc.pid)
                samples.append(
                    {
                        "time_utc": _utc_iso(),
                        "elapsed_s": now - start,
                        "pid": proc.pid,
                        "rss_mib": f"{rss_mib:.4f}",
                        "pss_mib": "" if pss_mib is None else f"{pss_mib:.4f}",
                        "uss_mib": "" if uss_mib is None else f"{uss_mib:.4f}",
                        "process_count": process_count,
                        "error": error,
                    }
                )
                stop.wait(args.interval)

        thread = threading.Thread(target=sampler, daemon=True)
        thread.start()
        return_code = proc.wait()
        stop.set()
        thread.join(timeout=max(args.interval * 2, 1.0))
        rss_mib, pss_mib, uss_mib, process_count, error = _process_tree_memory(proc.pid)
        samples.append(
            {
                "time_utc": _utc_iso(),
                "elapsed_s": time.time() - start,
                "pid": proc.pid,
                "rss_mib": f"{rss_mib:.4f}",
                "pss_mib": "" if pss_mib is None else f"{pss_mib:.4f}",
                "uss_mib": "" if uss_mib is None else f"{uss_mib:.4f}",
                "process_count": process_count,
                "error": error,
            }
        )
        log_f.write(f"EXIT_STATUS={return_code}\n")
        log_f.write(f"END_TIME={_local_iso()}\n")

    with args.csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "time_utc",
                "elapsed_s",
                "pid",
                "rss_mib",
                "pss_mib",
                "uss_mib",
                "process_count",
                "error",
            ],
        )
        writer.writeheader()
        writer.writerows(samples)

    numeric_rss = [float(s["rss_mib"]) for s in samples if s.get("process_count", 0)]
    numeric_pss = [
        float(s["pss_mib"])
        for s in samples
        if s.get("process_count", 0) and s.get("pss_mib") not in ("", None)
    ]
    numeric_uss = [
        float(s["uss_mib"])
        for s in samples
        if s.get("process_count", 0) and s.get("uss_mib") not in ("", None)
    ]
    process_counts = [int(s["process_count"]) for s in samples]
    summary = {
        "command": cmd,
        "return_code": return_code,
        "sample_interval_s": args.interval,
        "num_samples": len(samples),
        "peak_rss_mib": max(numeric_rss) if numeric_rss else None,
        "mean_rss_mib": mean(numeric_rss) if numeric_rss else None,
        "min_rss_mib": min(numeric_rss) if numeric_rss else None,
        "peak_pss_mib": max(numeric_pss) if numeric_pss else None,
        "mean_pss_mib": mean(numeric_pss) if numeric_pss else None,
        "min_pss_mib": min(numeric_pss) if numeric_pss else None,
        "peak_uss_mib": max(numeric_uss) if numeric_uss else None,
        "mean_uss_mib": mean(numeric_uss) if numeric_uss else None,
        "min_uss_mib": min(numeric_uss) if numeric_uss else None,
        "peak_process_count": max(process_counts) if process_counts else None,
    }
    args.summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
