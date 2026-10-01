#!/usr/bin/env python3
"""Acquire pinned official MELD splits and make aligned, 16 kHz mono manifests.

Only stdlib and the ffmpeg/curl executables are required. Dataset files stay under
--data-dir, which should be outside the source checkout. No pickle is executed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import tarfile
import time
import wave
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HF_REVISION = "9abc51ee7903424ffb971297608aa6d3d0de3bfa"
GITHUB_REVISION = "e8cedf27b5d2877e198332c957127e16eb214afe"
ARCHIVE_SHA256 = "a56b4407d574195cbce470d86f9c9d72fcfea59b0e34502ecd4babee4a5c613e"
ARCHIVE_SIZE = 10878146150
ARCHIVE_URL = f"https://huggingface.co/datasets/declare-lab/MELD/resolve/{HF_REVISION}/MELD.Raw.tar.gz?download=true"
CSV_SHA256 = {
    "train": "d2fa2d6529cf03cac2989efec05c9b27d8fd2f4c8fc5974c7ae88aa537fa02db",
    "dev": "2e89c6f8aa182d6f62f8c6331aece905ac7273ca4999660bfb5213e1d0370c1c",
    "test": "8d37103938f7067600839fe29d5a114a6cd1bcdafb75bec101e06464c5006888",
}
EXPECTED_COUNTS = {"train": 9989, "dev": 1109, "test": 2610}
LABELS = {"anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"}
CLIP_RE = re.compile(r"(?:final_videos_test)?dia\d+_utt\d+\.mp4\Z")
MAX_MODEL_AUDIO_SECONDS = 30.0


def sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def download(url: str, target: Path, checksum: str, expected_size: int | None = None) -> None:
    """Resume curl's .part file; authenticate completion with a pinned digest."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if (expected_size is None or target.stat().st_size == expected_size) and sha256(
            target
        ) == checksum:
            print(f"Verified cached {target.name}", flush=True)
            return
        raise RuntimeError(f"Checksum mismatch for cached {target}; remove it to redownload")
    partial = target.with_name(target.name + ".part")
    if partial.exists() and (expected_size is None or partial.stat().st_size == expected_size):
        if sha256(partial) == checksum:
            partial.replace(target)
            print(f"Verified completed partial {target.name}", flush=True)
            return
    if not (
        partial.exists() and expected_size is not None and partial.stat().st_size == expected_size
    ):
        print(f"Downloading {target.name} (resuming if present)", flush=True)
        subprocess.run(
            [
                "curl",
                "--fail",
                "--location",
                "--retry",
                "5",
                "--retry-delay",
                "5",
                "--continue-at",
                "-",
                "--output",
                str(partial),
                url,
            ],
            check=True,
        )
    if expected_size is not None and partial.stat().st_size != expected_size:
        raise RuntimeError(f"Unexpected download size: {partial}")
    if sha256(partial) != checksum:
        raise RuntimeError(f"Checksum mismatch for {partial}; remove it before retrying")
    partial.replace(target)
    print(f"Downloaded and verified {target.name}", flush=True)


def read_annotations(path: Path, split: str) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as source:
        rows = list(csv.DictReader(source))
    if len(rows) != EXPECTED_COUNTS[split]:
        raise ValueError(f"Expected {EXPECTED_COUNTS[split]} {split} rows, got {len(rows)}")
    records = []
    seen = set()
    for row in rows:
        dialogue_id, utterance_id = int(row["Dialogue_ID"]), int(row["Utterance_ID"])
        identifier = f"{split}/dia{dialogue_id}_utt{utterance_id}"
        if identifier in seen or row["Emotion"] not in LABELS:
            raise ValueError(f"Duplicate id or unknown emotion: {identifier}")
        seen.add(identifier)

        def timestamp_seconds(value: str) -> float:
            hours, minutes, seconds = value.replace(",", ".").split(":")
            return int(hours) * 3600 + int(minutes) * 60 + float(seconds)

        annotated_duration = timestamp_seconds(row["EndTime"]) - timestamp_seconds(row["StartTime"])
        records.append(
            {
                "annotated_duration_seconds": round(annotated_duration, 6),
                "id": identifier,
                "split": split,
                "dialogue_id": dialogue_id,
                "utterance_id": utterance_id,
                "text": row["Utterance"],
                "label": row["Emotion"],
                "speaker": row["Speaker"],
            }
        )
    return sorted(records, key=lambda r: (r["dialogue_id"], r["utterance_id"]))


def unpack_split_archives(archive: Path, destination: Path, splits: list[str]) -> dict[str, Path]:
    """Copy only expected regular members to generated paths, never extractall."""
    destination.mkdir(parents=True, exist_ok=True)
    targets = {split: destination / f"{split}.tar.gz" for split in splits}
    required = {split for split, path in targets.items() if not path.exists()}
    if not required:
        return targets
    found = set()
    # Stream the outer archive so a 10 GB member inventory isn't retained in RAM.
    with tarfile.open(archive, mode="r|gz") as source:
        for member in source:
            name = Path(member.name).name
            split = name.removesuffix(".tar.gz")
            if split not in required or name != f"{split}.tar.gz":
                continue
            if not member.isfile() or split in found:
                raise ValueError(f"Invalid or repeated split archive: {member.name}")
            found.add(split)
            target = targets[split]
            temporary = target.with_suffix(target.suffix + ".tmp")
            print(f"Unpacking {name}", flush=True)
            stream = source.extractfile(member)
            if stream is None:
                raise ValueError(f"Unreadable archive member: {member.name}")
            with stream, temporary.open("wb") as output:
                shutil.copyfileobj(stream, output, length=1024 * 1024)
            if temporary.stat().st_size != member.size:
                raise ValueError(f"Truncated archive member: {member.name}")
            temporary.replace(target)
    if found != required:
        raise ValueError(f"Missing split archives: {sorted(required - found)}")
    return targets


def unpack_clips(archive: Path, destination: Path) -> set[str]:
    """Flatten known MP4 names; no archive-supplied path is ever used for writes."""
    destination.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    with tarfile.open(archive, mode="r|gz") as source:
        for member in source:
            name = Path(member.name).name
            if not CLIP_RE.fullmatch(name):
                continue
            if not member.isfile() or name in seen:
                raise ValueError(f"Invalid or duplicate clip name: {member.name}")
            seen.add(name)
            target = destination / name
            if target.exists() and target.stat().st_size == member.size:
                continue
            temporary = target.with_suffix(".mp4.tmp")
            stream = source.extractfile(member)
            if stream is None:
                raise ValueError(f"Unreadable clip: {member.name}")
            with stream, temporary.open("wb") as output:
                shutil.copyfileobj(stream, output)
            if temporary.stat().st_size != member.size:
                raise ValueError(f"Truncated clip: {member.name}")
            temporary.replace(target)
    return seen


def inspect_wav(path: Path) -> dict:
    with wave.open(str(path), "rb") as source:
        if (
            source.getnchannels() != 1
            or source.getframerate() != 16000
            or source.getsampwidth() != 2
        ):
            raise ValueError("WAV must be mono 16 kHz PCM16")
        frames = source.getnframes()
        if frames < 160:
            raise ValueError("Audio has fewer than 10 ms of samples")
        if len(source.readframes(frames)) != frames * 2:
            raise ValueError("Truncated PCM samples")
    return {"sample_rate": 16000, "audio_samples": frames, "duration_seconds": frames / 16000}


def convert_clip(record: dict, video_dir: Path, audio_dir: Path, ffmpeg: str) -> dict:
    result = dict(record)
    stem = record["id"].split("/")[1]
    target = audio_dir / f"{stem}.wav"
    result["audio_path"] = str(target.resolve())
    video = video_dir / f"{stem}.mp4"
    alternate = video_dir / f"final_videos_test{stem}.mp4"
    if record["split"] == "test" and alternate.exists():
        video = alternate
    result["audio_source_filename"] = video.name
    source_marker = target.with_suffix(".source.json")
    try:
        if not video.exists():
            raise FileNotFoundError("Official archive has no matching MP4")
        source_digest = sha256(video)
        source_identity = {"filename": video.name, "sha256": source_digest}
        result["audio_source_sha256"] = source_digest
        cached_identity = json.loads(source_marker.read_text()) if source_marker.exists() else None
        if target.exists() and cached_identity == source_identity:
            try:
                info = inspect_wav(target)
            except (ValueError, wave.Error, EOFError):
                target.unlink()
            else:
                result.update(info, audio_available=True, audio_sha256=sha256(target))
                return result
        temporary = target.with_suffix(".tmp.wav")
        process = subprocess.run(
            [
                ffmpeg,
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-threads",
                "1",
                "-y",
                "-i",
                str(video),
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-c:a",
                "pcm_s16le",
                str(temporary),
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if process.returncode:
            temporary.unlink(missing_ok=True)
            raise ValueError(process.stderr.strip()[-1000:] or f"ffmpeg exit {process.returncode}")
        info = inspect_wav(temporary)
        temporary.replace(target)
        atomic_json(source_marker, source_identity)
        result.update(info, audio_available=True, audio_sha256=sha256(target))
        if process.stderr.strip():
            result["audio_warning"] = process.stderr.strip()[-1000:]
    except (OSError, ValueError, wave.Error, EOFError, subprocess.TimeoutExpired) as error:
        result["audio_available"] = False
        result["audio_error"] = str(error)
    return result


def mark_audio_eligibility(record: dict) -> dict:
    """Use the same 30-second utterance limit for learning and live inference."""
    usable = (
        record.get("audio_available", False)
        and record.get("duration_seconds", 0) <= MAX_MODEL_AUDIO_SECONDS
    )
    record["audio_usable"] = bool(usable)
    if not record.get("audio_available", False):
        record["audio_exclusion_reason"] = "missing_or_corrupt_source"
    elif not usable:
        record["audio_exclusion_reason"] = "exceeds_30_second_runtime_limit"
    if record.get("audio_available") and "annotated_duration_seconds" in record:
        record["audio_duration_delta_seconds"] = round(
            record["duration_seconds"] - record["annotated_duration_seconds"], 6
        )
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--splits", nargs="+", choices=list(EXPECTED_COUNTS), default=["train", "dev", "test"]
    )
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 4))
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Return success despite explicit audio failures; no rows are dropped",
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    if shutil.which(args.ffmpeg) is None or shutil.which("curl") is None:
        parser.error("Install ffmpeg and curl before preparing MELD")
    data_dir = args.data_dir.expanduser().resolve()
    downloads = data_dir / "downloads"
    data_dir.mkdir(parents=True, exist_ok=True)
    sources = {
        "huggingface_revision": HF_REVISION,
        "github_revision": GITHUB_REVISION,
        "archive": {"url": ARCHIVE_URL, "sha256": ARCHIVE_SHA256, "bytes": ARCHIVE_SIZE},
        "annotations": {},
        "conversion": {
            "ffmpeg_version": subprocess.run(
                [args.ffmpeg, "-version"], capture_output=True, text=True, check=True
            ).stdout.splitlines()[0],
            "sample_rate": 16000,
            "channels": 1,
            "encoding": "pcm_s16le",
            "trim": None,
        },
    }
    rows_by_split = {}
    for split in args.splits:
        url = f"https://raw.githubusercontent.com/declare-lab/MELD/{GITHUB_REVISION}/data/MELD/{split}_sent_emo.csv"
        target = downloads / f"{split}_sent_emo.csv"
        download(url, target, CSV_SHA256[split])
        rows_by_split[split] = read_annotations(target, split)
        sources["annotations"][split] = {"url": url, "sha256": CSV_SHA256[split]}
    # Preserve full source provenance when a later run prepares only one split.
    for split, checksum in CSV_SHA256.items():
        sources["annotations"].setdefault(
            split,
            {
                "url": f"https://raw.githubusercontent.com/declare-lab/MELD/{GITHUB_REVISION}/data/MELD/{split}_sent_emo.csv",
                "sha256": checksum,
            },
        )
    atomic_json(data_dir / "provenance.json", sources)
    archive = downloads / "MELD.Raw.tar.gz"
    download(ARCHIVE_URL, archive, ARCHIVE_SHA256, ARCHIVE_SIZE)
    split_archives = unpack_split_archives(archive, downloads / "splits", args.splits)
    summary_path = data_dir / "data_summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    failures = 0
    for split in args.splits:
        started = time.monotonic()
        video_dir = data_dir / "video" / split
        audio_dir = data_dir / "audio" / split
        audio_dir.mkdir(parents=True, exist_ok=True)
        print(f"Preparing {split}: {len(rows_by_split[split])} annotated utterances", flush=True)
        present = unpack_clips(split_archives[split], video_dir)
        records = []
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            pending = [
                pool.submit(convert_clip, row, video_dir, audio_dir, args.ffmpeg)
                for row in rows_by_split[split]
            ]
            for future in as_completed(pending):
                records.append(mark_audio_eligibility(future.result()))
                if len(records) % 500 == 0:
                    print(f"{split}: {len(records)}/{len(pending)} decoded", flush=True)
        records.sort(key=lambda r: (r["dialogue_id"], r["utterance_id"]))
        manifests = data_dir / "manifests"
        manifests.mkdir(exist_ok=True)
        manifest = manifests / f"{split}.jsonl"
        temporary = manifest.with_suffix(".jsonl.tmp")
        with temporary.open("w") as output:
            for record in records:
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
        temporary.replace(manifest)
        missing = [
            {"id": r["id"], "error": r["audio_error"]} for r in records if not r["audio_available"]
        ]
        warnings = [
            {"id": r["id"], "warning": r["audio_warning"]}
            for r in records
            if r.get("audio_warning")
        ]
        durations = [r["duration_seconds"] for r in records if r["audio_available"]]
        expected = {r["id"].split("/")[1] + ".mp4" for r in records}
        summary[split] = {
            "annotation_count": len(records),
            "audio_available": len(durations),
            "audio_usable": sum(r["audio_usable"] for r in records),
            "audio_exclusions": [
                {"id": r["id"], "reason": r["audio_exclusion_reason"]}
                for r in records
                if not r["audio_usable"]
            ],
            "duration_mismatches_over_one_second": [
                {"id": r["id"], "delta_seconds": r["audio_duration_delta_seconds"]}
                for r in records
                if abs(r.get("audio_duration_delta_seconds", 0)) > 1
            ],
            "dialogues": len({r["dialogue_id"] for r in records}),
            "labels": dict(sorted(Counter(r["label"] for r in records).items())),
            "audio_failures": missing,
            "decode_warnings": warnings,
            "extra_unannotated_clips": sorted(
                name for name in present if name.removeprefix("final_videos_test") not in expected
            ),
            "preferred_test_variants": [
                r["id"]
                for r in records
                if r.get("audio_source_filename", "").startswith("final_videos_test")
            ],
            "duration_seconds": {
                "total": sum(durations),
                "min": min(durations, default=0),
                "max": max(durations, default=0),
            },
            "manifest_sha256": sha256(manifest),
            "prepare_seconds": round(time.monotonic() - started, 2),
        }
        atomic_json(summary_path, summary)
        failures += len(missing)
        print(
            f"Ready: {manifest} ({len(durations)}/{len(records)} audio available; {len(missing)} failures)",
            flush=True,
        )
    print(f"Audit report: {summary_path}", flush=True)
    return 1 if failures and not args.allow_missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
