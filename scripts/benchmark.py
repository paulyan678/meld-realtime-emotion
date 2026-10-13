"""Measure complete local inference with socket connections blocked and no feature cache."""

import argparse
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import numpy as np
import psutil
import soundfile as sf

from meld_emotion.constants import AUDIO_MODEL, AUDIO_REVISION, TEXT_MODEL, TEXT_REVISION
from meld_emotion.features import read_manifest
from meld_emotion.inference import EmotionEngine


def blocked(*args, **kwargs):
    raise RuntimeError("Network access blocked during offline benchmark")


@contextmanager
def block_network():
    originals = socket.socket.connect, socket.socket.connect_ex, socket.create_connection
    socket.socket.connect = socket.socket.connect_ex = socket.create_connection = blocked
    try:
        yield
    finally:
        socket.socket.connect, socket.socket.connect_ex, socket.create_connection = originals


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command_output(command, cwd=None):
    try:
        return subprocess.run(
            command, cwd=cwd, capture_output=True, text=True, check=True, timeout=10
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def source_fingerprint(root):
    root = Path(root)
    sources = sorted(
        path
        for base in (root / "src", root / "scripts")
        for path in base.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
    )
    sources += [
        path
        for path in (root / "pyproject.toml", root / "requirements-lock-macos.txt")
        if path.is_file()
    ]
    status = command_output(["git", "status", "--porcelain"], cwd=root)
    return {
        "revision": command_output(["git", "rev-parse", "HEAD"], cwd=root),
        "worktree_dirty": None if status is None else bool(status),
        "source_sha256": {str(path.relative_to(root)): sha256(path) for path in sources},
    }


def validate_loaded_sources(root):
    expected = (Path(root) / "src").resolve()
    for name, module in tuple(sys.modules.items()):
        if name != "meld_emotion" and not name.startswith("meld_emotion."):
            continue
        origin = getattr(module, "__file__", None)
        if origin is None or not Path(origin).resolve().is_relative_to(expected):
            raise RuntimeError(
                f"Loaded {name} does not come from this checkout's src directory. "
                "Reinstall this checkout with python -m pip install -e . "
                "or set PYTHONPATH=$PWD/src before running the benchmark."
            )


def model_fingerprint(model_dir):
    from huggingface_hub import hf_hub_download

    encoders = {}
    for name, repository, revision in (
        ("text", TEXT_MODEL, TEXT_REVISION),
        ("audio_and_asr", AUDIO_MODEL, AUDIO_REVISION),
    ):
        cached = hf_hub_download(
            repository, filename="model.safetensors", revision=revision, local_files_only=True
        )
        encoders[name] = {
            "model": repository,
            "revision": revision,
            "weights_sha256": sha256(cached),
        }
    return {
        "task_files_sha256": {
            name: sha256(Path(model_dir) / name) for name in ("heads.npz", "metadata.json")
        },
        "pretrained": encoders,
    }


def hardware(device):
    chip = platform.processor() or None
    model = None
    if platform.system() == "Darwin":
        chip = command_output(["sysctl", "-n", "machdep.cpu.brand_string"]) or chip
        model = command_output(["sysctl", "-n", "hw.model"])
    elif platform.system() == "Linux":
        cpuinfo = Path("/proc/cpuinfo")
        if cpuinfo.exists():
            chip = next(
                (
                    line.split(":", 1)[1].strip()
                    for line in cpuinfo.read_text().splitlines()
                    if line.startswith("model name")
                ),
                chip,
            )
    result = {
        "device": device,
        "chip": chip,
        "model": model,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "physical_memory_bytes": psutil.virtual_memory().total,
        "cpu_logical_cores": psutil.cpu_count(),
        "torch_threads": int(os.environ.get("MELD_TORCH_THREADS", "4")),
    }
    if device == "cuda":
        import torch

        result["gpu"] = torch.cuda.get_device_name()
    return result


def attempt(engine, row, automatic):
    """Time WAV disk decode through the final prediction, including failed attempts."""
    started = time.perf_counter()
    trace = {"id": row["id"], "status": "failed"}
    try:
        audio, rate = sf.read(row["audio_path"], dtype="float32")
        trace["duration_seconds"] = len(audio) / rate
        text = row["text"]
        if automatic:
            transcription_started = time.perf_counter()
            try:
                text = engine.transcribe(audio, rate)
            finally:
                trace["transcription_ms"] = (time.perf_counter() - transcription_started) * 1000
        result = engine.predict(text, audio, rate)
        trace.update(
            status="complete",
            emotion=result["emotion"],
            confidence=result["confidence"],
            modality_status=result["modality_status"],
        )
    except Exception as error:
        trace.update(error_type=type(error).__name__, error=str(error))
    trace["wall_ms"] = (time.perf_counter() - started) * 1000
    return trace


def run_benchmark(args):
    root = Path(__file__).resolve().parents[1]
    validate_loaded_sources(root)
    rows = read_manifest(args.manifest)
    total_rows = len(rows)
    rows = [
        row
        for row in rows
        if row.get("audio_available", True) and 0.1 <= row.get("duration_seconds", 0) <= 30
    ]
    if not rows or args.samples < 1:
        raise ValueError("Need positive sample count and eligible 0.1–30s audio rows")
    eligible_rows = len(rows)
    rows = [rows[i] for i in np.linspace(0, len(rows) - 1, min(args.samples, len(rows)), dtype=int)]
    with block_network():
        provenance = {
            "source": source_fingerprint(root),
            "manifest_sha256": sha256(args.manifest),
            "models": model_fingerprint(args.model_dir),
        }
        host = hardware(args.device)
        process = psutil.Process()
        peak = [process.memory_info().rss]
        stop = threading.Event()

        def monitor():
            while not stop.wait(0.01):
                peak[0] = max(peak[0], process.memory_info().rss)

        thread = threading.Thread(target=monitor, daemon=True)
        thread.start()
        try:
            began = time.perf_counter()
            engine = EmotionEngine(args.model_dir, device=args.device, local_files_only=True)
            if args.automatic:
                engine.encoders.enable_transcription()
                loaded_asr = sum(p.numel() for p in engine.encoders.asr_head.parameters())
                if loaded_asr != engine.parameter_counts["asr_ctc_head"]:
                    raise ValueError("Loaded ASR parameters differ from the reported count")
            load_ms = (time.perf_counter() - began) * 1000
            counts = dict(engine.parameter_counts)
            cold = attempt(engine, rows[0], args.automatic)
            cold["including_model_load_ms"] = (time.perf_counter() - began) * 1000
            traces = [attempt(engine, row, args.automatic) for row in rows]
        finally:
            peak[0] = max(peak[0], process.memory_info().rss)
            stop.set()
            thread.join()

        times = [trace["wall_ms"] for trace in traces if trace["status"] == "complete"]
        report = {
            "schema_version": 2,
            "recorded_at_utc": datetime.now(UTC).isoformat(),
            **host,
            "provenance": provenance,
            "network_socket_connections": "connect, connect_ex, create_connection blocked",
            "feature_cache": False,
            "transcript_source": "automatic" if args.automatic else "provided",
            "asr_head_loaded": args.automatic,
            "model_load_ms": load_ms,
            "cold_attempt": cold,
            "cold_start_scope": (
                "Fresh model initialization; checkpoint files were already read for fingerprinting. "
                "OS filesystem cache is uncontrolled. This is not a cold-disk measurement."
            ),
            "warmup_completed": cold["status"] == "complete",
            "samples": len(traces),
            "completed": len(times),
            "failed": len(traces) - len(times),
            "latency_population": (
                "successful measured attempts after one cold attempt; cold reported separately. "
                "If warmup_completed is false, the first successful sample may include warmup."
            ),
            "warm_p50_ms": float(np.percentile(times, 50)) if times else None,
            "warm_p95_ms": float(np.percentile(times, 95)) if times else None,
            "warm_max_ms": max(times) if times else None,
            "source_rows": total_rows,
            "eligible_rows": eligible_rows,
            "excluded_rows": total_rows - eligible_rows,
            "sample_split": rows[0]["split"],
            "sample_selection": "fixed stride across eligible rows, independent of outputs",
            "peak_process_rss_bytes_sampled_10ms": peak[0],
            "memory_boundary": "model and ASR loading, cold attempt, then all warm attempts",
            "parameter_counts": counts,
            "timing_boundary": (
                "WAV disk decode through "
                + ("local transcription and " if args.automatic else "")
                + "emotion and template reply; excludes recording and transport. "
                "Model loading is separate and also included in cold including_model_load_ms. "
                "The first selected clip is attempted once cold and again in the warm sample."
            ),
            "traces": traces,
        }
        if args.device == "mps":
            import torch

            report["mps_current_allocated_bytes_at_end"] = torch.mps.current_allocated_memory()
            report["mps_driver_allocated_bytes_at_end"] = torch.mps.driver_allocated_memory()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--model-dir", default="artifacts/model")
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument(
        "--automatic", action="store_true", help="Transcribe audio before prediction"
    )
    parser.add_argument("--output", default="reports/runtime.json")
    args = parser.parse_args()
    report = run_benchmark(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "traces"}, indent=2))


if __name__ == "__main__":
    main()
