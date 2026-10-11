"""Exercise real HTTP and successive streamed utterances against a local server."""

import argparse
import base64
import hashlib
import json
import time
from pathlib import Path

import httpx
import numpy as np
import soundfile as sf
from websockets.sync.client import connect

from meld_emotion.features import read_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--output", default="reports/interaction.json")
    parser.add_argument("--automatic", action="store_true", help="Send audio without transcripts")
    args = parser.parse_args()
    rows = [
        r
        for r in read_manifest(args.manifest)
        if r.get("audio_available", True) and 0.1 <= r.get("duration_seconds", 0) <= 30
    ]
    if not rows or args.samples < 1:
        parser.error("Need eligible audio and a positive sample count")
    selected = [
        rows[i] for i in np.linspace(0, len(rows) - 1, min(args.samples, len(rows)), dtype=int)
    ]
    with httpx.Client(timeout=120) as client:
        before = client.get(args.url + "/health").json()
        first = selected[0]
        with open(first["audio_path"], "rb") as file:
            start = time.perf_counter()
            response = client.post(
                args.url + "/predict",
                data={} if args.automatic else {"text": first["text"]},
                files={"audio": ("utterance.wav", file, "audio/wav")},
            )
            cold_http_ms = (time.perf_counter() - start) * 1000
        response.raise_for_status()
        http_result = response.json()
        traces = []
        full_example = None
        clock = 0
        with connect(
            args.url.replace("http://", "ws://").replace("https://", "wss://") + "/stream",
            max_size=1_000_000,
            open_timeout=10,
        ) as socket:
            assert json.loads(socket.recv())["type"] == "ready"

            def send(kind, identifier, **fields):
                nonlocal clock
                clock += 1
                socket.send(
                    json.dumps(
                        {"type": kind, "utterance_id": identifier, "timestamp_ms": clock, **fields}
                    )
                )
                result = json.loads(socket.recv())
                while result["type"] == "processing":
                    assert result["utterance_id"] == identifier
                    result = json.loads(socket.recv())
                if result["type"] == "error" and not args.automatic:
                    raise RuntimeError(result)
                return result

            for row in selected:
                audio, sr = sf.read(row["audio_path"], dtype="float32")
                assert send("start", row["id"], sample_rate=sr)["type"] == "ack"
                if not args.automatic:
                    assert send("text", row["id"], text=row["text"])["type"] == "ack"
                chunk_count = 0
                for sequence, offset in enumerate(range(0, len(audio), 4096)):
                    chunk = audio[offset : offset + 4096].astype("<f4").tobytes()
                    assert (
                        send(
                            "audio",
                            row["id"],
                            sequence=sequence,
                            encoding="pcm_f32le",
                            data=base64.b64encode(chunk).decode(),
                        )["type"]
                        == "ack"
                    )
                    chunk_count += 1
                started = time.perf_counter()
                result = send("end", row["id"])
                elapsed = (time.perf_counter() - started) * 1000
                if args.automatic and result["type"] == "error":
                    assert result["utterance_id"] == row["id"]
                    # Unrecognized speech is an explicit, recoverable failure, not a prediction.
                    traces.append(
                        {
                            "id": row["id"],
                            "duration_seconds": len(audio) / sr,
                            "audio_chunks": chunk_count,
                            "post_end_ms": elapsed,
                            "status": "error",
                            "http_status": result["status"],
                            "message": result["message"],
                        }
                    )
                    continue
                assert result["type"] == "result" and result["utterance_id"] == row["id"]
                assert abs(sum(result["probabilities"].values()) - 1) < 1e-6
                assert result["response"] and result["modality_status"] == "text_and_audio"
                if args.automatic:
                    assert result["transcript_source"] == "automatic" and result["transcript"]
                traces.append(
                    {
                        "id": row["id"],
                        "duration_seconds": len(audio) / sr,
                        "audio_chunks": chunk_count,
                        "post_end_ms": elapsed,
                        "emotion": result["emotion"],
                        "status": "complete",
                        "transcription_ms": result.get("transcription_ms", 0),
                    }
                )
                if full_example is None:
                    assert result["emotion"] == http_result["emotion"]
                    assert np.allclose(
                        list(result["probabilities"].values()),
                        list(http_result["probabilities"].values()),
                        atol=1e-6,
                    )
                    full_example = {
                        "id": row["id"],
                        "reference_transcript": row["text"],
                        "audio_sha256": hashlib.sha256(
                            Path(row["audio_path"]).read_bytes()
                        ).hexdigest(),
                        "ground_truth_for_reference_only": row["label"],
                        "result": result,
                    }
        after = client.get(args.url + "/health").json()
    latencies = [t["post_end_ms"] for t in traces if t["status"] == "complete"]
    report = {
        "transport": "HTTP WAV upload and persistent WebSocket PCM stream",
        "timing_boundary": "client sends end event to receives complete emotion and response JSON, localhost",
        "samples": len(traces),
        "transcript_source": "automatic" if args.automatic else "provided",
        "completed": len(latencies),
        "failed": len(traces) - len(latencies),
        "latency_population": "completed requests only; failures remain in traces",
        "sample_selection": "fixed stride across eligible development rows, no output selection",
        "initial_health": before,
        "final_health": after,
        "first_http_including_lazy_load_ms": cold_http_ms,
        "warm_post_end_p50_ms": float(np.percentile(latencies, 50)),
        "warm_post_end_p95_ms": float(np.percentile(latencies, 95)),
        "warm_post_end_max_ms": max(latencies),
        "example": full_example,
        "traces": traces,
    }
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in {"example", "traces"}}, indent=2))


if __name__ == "__main__":
    main()
