"""Resumable frozen feature extraction. No labels influence the encoders."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from .constants import AUDIO_REVISION, LABELS, TEXT_REVISION
from .encoders import Encoders


def read_manifest(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate manifest IDs")
    for row in rows:
        if row["label"] not in LABELS:
            raise ValueError(f"Unknown label: {row['label']}")
    return rows


def extract(manifest, output, cache, device="cpu"):
    rows = read_manifest(manifest)
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    encoder = Encoders(device=device)
    texts = encoder.text([r["text"] for r in rows[:1]])  # warm-up
    del texts
    text_vectors = np.concatenate(
        [encoder.text([r["text"] for r in rows[i : i + 64]]) for i in range(0, len(rows), 64)]
    )
    audio_vectors = []
    durations = []
    rms = []
    available = []
    start = time.perf_counter()
    for index, row in enumerate(rows):
        if not row.get("audio_available", True):
            audio_vectors.append(np.zeros(1536, dtype=np.float32))
            durations.append(0.0)
            rms.append(0.0)
            available.append(False)
            print(f"Explicit missing-audio row: {row['id']}", flush=True)
            continue
        available.append(True)
        path = Path(row["audio_path"])
        # Content hash prevents stale features if preparation or media change.
        key = hashlib.sha256(
            path.read_bytes() + AUDIO_REVISION.encode() + b"meanstd6s-v1"
        ).hexdigest()
        cached = cache / f"{key}.npz"
        audio, sr = sf.read(path, dtype="float32")
        if cached.exists():
            with np.load(cached, allow_pickle=False) as data:
                vector = data["audio"]
        else:
            vector = encoder.audio(audio, sr)
            np.savez_compressed(cached, audio=vector)
        audio_vectors.append(vector)
        durations.append(len(audio) / sr)
        rms.append(float(np.std(audio)))  # RMS after removing DC offset
        if (index + 1) % 100 == 0 or index + 1 == len(rows):
            elapsed = time.perf_counter() - start
            print(f"{index + 1}/{len(rows)} audio features; {elapsed:.1f}s", flush=True)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        text=text_vectors,
        audio=np.stack(audio_vectors),
        labels=np.array([LABELS.index(r["label"]) for r in rows]),
        ids=np.array([r["id"] for r in rows]),
        durations=np.array(durations),
        rms=np.array(rms),
        text_revision=np.array(TEXT_REVISION),
        audio_revision=np.array(AUDIO_REVISION),
        feature_version=np.array("meanstd6s-v1"),
        audio_available=np.array(available, dtype=bool),
    )
    output.with_suffix(".json").write_text(
        json.dumps(
            {
                "manifest_sha256": hashlib.sha256(Path(manifest).read_bytes()).hexdigest(),
                "rows": len(rows),
                "text_revision": TEXT_REVISION,
                "audio_revision": AUDIO_REVISION,
                "available_audio_rows": sum(available),
                "device": device,
                "parameter_counts": encoder.parameter_counts,
                "audio_extraction_elapsed_seconds": time.perf_counter() - start,
            },
            indent=2,
        )
        + "\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    extract(args.manifest, args.output, args.cache, args.device)


if __name__ == "__main__":
    main()
