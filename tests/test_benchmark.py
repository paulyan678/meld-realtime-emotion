"""Benchmark accounting tests use local synthetic audio and no pretrained models."""

import importlib.util
import json
import socket
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def benchmark():
    path = Path(__file__).resolve().parents[1] / "scripts" / "benchmark.py"
    spec = importlib.util.spec_from_file_location("benchmark_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def setup_run(benchmark, monkeypatch, tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    rows = [
        {
            "id": f"dev/{i}",
            "audio_path": str(i),
            "text": "reference",
            "duration_seconds": 1,
            "split": "dev",
        }
        for i in range(3)
    ]
    manifest.write_text("\n".join(json.dumps(row) for row in rows))
    monkeypatch.setattr(benchmark, "read_manifest", lambda path: rows)
    monkeypatch.setattr(benchmark, "model_fingerprint", lambda path: {"test": "synthetic"})
    monkeypatch.setattr(benchmark, "source_fingerprint", lambda path: {"revision": "test"})
    monkeypatch.setattr(benchmark, "hardware", lambda device: {"device": device})
    monkeypatch.setattr(
        benchmark.sf, "read", lambda path, dtype: (np.full(16000, int(path)), 16000)
    )
    state = {"calls": [], "failed_ids": {0}, "asr_loaded": False}

    class FakeEngine:
        parameter_counts = {"text_encoder": 3, "audio_encoder": 4, "asr_ctc_head": 2, "total": 9}

        def __init__(self, model_dir, device, local_files_only):
            assert local_files_only is True
            with pytest.raises(RuntimeError, match="Network access blocked"):
                socket.create_connection(("example.invalid", 443))
            self.encoders = SimpleNamespace(
                enable_transcription=self.enable,
                asr_head=SimpleNamespace(parameters=lambda: [SimpleNamespace(numel=lambda: 2)]),
            )

        def enable(self):
            state["asr_loaded"] = True

        def transcribe(self, audio, rate):
            assert state["asr_loaded"]
            if int(audio[0]) in state["failed_ids"]:
                raise ValueError("No speech recognized. Please speak clearly and try again.")
            return "recognized words"

        def predict(self, text, audio, rate):
            state["calls"].append((int(audio[0]), text))
            return {"emotion": "joy", "confidence": 0.6, "modality_status": "text_and_audio"}

    monkeypatch.setattr(benchmark, "EmotionEngine", FakeEngine)
    args = SimpleNamespace(
        manifest=manifest, model_dir=tmp_path, device="cpu", samples=3, automatic=True
    )
    return args, state


def test_no_speech_failures_keep_cold_and_warm_denominators(benchmark, setup_run):
    args, state = setup_run
    original_connect = socket.socket.connect
    report = benchmark.run_benchmark(args)
    assert report["cold_attempt"]["status"] == "failed"
    assert report["cold_attempt"]["error_type"] == "ValueError"
    assert report["warmup_completed"] is False
    assert report["cold_attempt"]["including_model_load_ms"] >= report["cold_attempt"]["wall_ms"]
    assert (report["samples"], report["completed"], report["failed"]) == (3, 2, 1)
    assert [trace["status"] for trace in report["traces"]] == ["failed", "complete", "complete"]
    assert state["calls"] == [(1, "recognized words"), (2, "recognized words")]
    completed = [trace["wall_ms"] for trace in report["traces"] if trace["status"] == "complete"]
    assert report["warm_p50_ms"] == float(np.percentile(completed, 50))
    assert report["parameter_counts"]["asr_ctc_head"] == 2
    assert socket.socket.connect is original_connect


def test_all_failed_attempts_are_reported_without_percentile_errors(benchmark, setup_run):
    args, state = setup_run
    state["failed_ids"] = {0, 1, 2}
    report = benchmark.run_benchmark(args)
    assert (report["samples"], report["completed"], report["failed"]) == (3, 0, 3)
    assert report["cold_attempt"]["status"] == "failed"
    assert report["warm_p50_ms"] is report["warm_p95_ms"] is report["warm_max_ms"] is None
    assert state["calls"] == []
    assert len(report["traces"]) == 3


def test_provided_transcripts_do_not_load_or_run_asr(benchmark, setup_run):
    args, state = setup_run
    args.automatic = False
    report = benchmark.run_benchmark(args)
    assert report["transcript_source"] == "provided"
    assert report["completed"] == 3 and report["failed"] == 0
    assert report["cold_attempt"]["status"] == "complete"
    assert report["warmup_completed"] is True
    assert not state["asr_loaded"]
    assert state["calls"] == [
        (0, "reference"),
        (0, "reference"),
        (1, "reference"),
        (2, "reference"),
    ]


def test_socket_blocking_covers_connect_ex_and_restores_on_error(benchmark):
    original = socket.socket.connect, socket.socket.connect_ex, socket.create_connection
    with pytest.raises(ValueError, match="stop"):
        with benchmark.block_network(), socket.socket() as client:
            for method in (client.connect, client.connect_ex, socket.create_connection):
                with pytest.raises(RuntimeError, match="Network access blocked"):
                    method(("127.0.0.1", 1))
            raise ValueError("stop")
    assert (socket.socket.connect, socket.socket.connect_ex, socket.create_connection) == original


def test_source_hash_detects_uncommitted_changes_without_absolute_paths(benchmark, tmp_path):
    source = tmp_path / "src" / "demo.py"
    source.parent.mkdir()
    source.write_text("value = 1\n")
    first = benchmark.source_fingerprint(tmp_path)
    source.write_text("value = 2\n")
    second = benchmark.source_fingerprint(tmp_path)
    assert set(first["source_sha256"]) == {"src/demo.py"}
    assert first["source_sha256"] != second["source_sha256"]
    assert first["revision"] is None and first["worktree_dirty"] is None


def test_rejects_imports_from_another_editable_checkout(benchmark, monkeypatch, tmp_path):
    root = Path(benchmark.__file__).resolve().parents[1]
    benchmark.validate_loaded_sources(root)
    imported = benchmark.sys.modules["meld_emotion.inference"]
    monkeypatch.setattr(imported, "__file__", str(tmp_path / "other_checkout" / "inference.py"))
    with pytest.raises(RuntimeError, match="Reinstall this checkout.*PYTHONPATH"):
        benchmark.validate_loaded_sources(root)
