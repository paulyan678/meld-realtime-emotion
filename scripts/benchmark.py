"""Measure raw-input runtime with network sockets disabled (no feature cache)."""

import argparse
import json
import os
import platform
import socket
import threading
import time
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import numpy as np
import psutil
import soundfile as sf

from meld_emotion.features import read_manifest
from meld_emotion.inference import EmotionEngine


def blocked(*args, **kwargs):
    raise RuntimeError("Network access blocked during offline benchmark")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--model-dir", default="artifacts/model")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--output", default="reports/runtime.json")
    args = parser.parse_args()
    socket.socket.connect = blocked
    socket.create_connection = blocked
    rows = read_manifest(args.manifest)
    total_rows = len(rows)
    rows = [
        r
        for r in rows
        if r.get("audio_available", True) and 0.1 <= r.get("duration_seconds", 0) <= 30
    ]
    if not rows or args.samples < 1:
        parser.error("Need positive sample count and eligible 0.1–30s audio rows")
    eligible_rows = len(rows)
    # Fixed stride sample across development, selected without model outputs.
    rows = [rows[i] for i in np.linspace(0, len(rows) - 1, min(args.samples, len(rows)), dtype=int)]
    process = psutil.Process()
    peak = [process.memory_info().rss]
    stop = threading.Event()

    def monitor():
        while not stop.wait(0.01):
            peak[0] = max(peak[0], process.memory_info().rss)

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    begin = time.perf_counter()
    engine = EmotionEngine(args.model_dir, device=args.device)
    load_ms = (time.perf_counter() - begin) * 1000
    audio, sr = sf.read(rows[0]["audio_path"], dtype="float32")
    first = engine.predict(rows[0]["text"], audio, sr)
    traces = []
    for row in rows:
        audio, sr = sf.read(row["audio_path"], dtype="float32")
        if len(audio) / sr > 30:
            continue
        started = time.perf_counter()
        result = engine.predict(row["text"], audio, sr)
        traces.append(
            {
                "id": row["id"],
                "duration_seconds": len(audio) / sr,
                "wall_ms": (time.perf_counter() - started) * 1000,
                "emotion": result["emotion"],
                "modality_status": result["modality_status"],
            }
        )
    stop.set()
    thread.join()
    times = [row["wall_ms"] for row in traces]
    report = {
        "device": args.device,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "physical_memory_bytes": psutil.virtual_memory().total,
        "cpu_logical_cores": psutil.cpu_count(),
        "torch_threads": int(os.environ.get("MELD_TORCH_THREADS", "4")),
        "network_socket_connections": "blocked",
        "feature_cache": False,
        "encoder_load_ms": load_ms,
        "cold_first_prediction_ms": first["latency_ms"],
        "samples": len(traces),
        "warm_p50_ms": float(np.percentile(times, 50)),
        "source_rows": total_rows,
        "eligible_rows": eligible_rows,
        "excluded_rows": total_rows - eligible_rows,
        "sample_split": rows[0]["split"],
        "warm_p95_ms": float(np.percentile(times, 95)),
        "warm_max_ms": max(times),
        "peak_process_rss_bytes_sampled_10ms": peak[0],
        "parameter_counts": engine.parameter_counts,
        "timing_boundary": "in-memory transcript and decoded audio to final tags and response; excludes recording, disk decode, transport",
        "traces": traces,
    }
    if args.device == "mps":
        import torch

        report["mps_current_allocated_bytes"] = torch.mps.current_allocated_memory()
        report["mps_driver_allocated_bytes_at_end"] = torch.mps.driver_allocated_memory()
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "traces"}, indent=2))


if __name__ == "__main__":
    main()
