"""Data-integrity regressions using synthetic archives and an FFmpeg boundary fake.

These tests need neither MELD downloads nor FFmpeg. The decoder fake writes a
valid PCM WAV with source-dependent samples so incorrect source selection or
stale cache reuse is observable in the resulting artifact.
"""

import importlib.util
import io
import os
import struct
import subprocess
import tarfile
import wave
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def pipeline():
    path = Path(__file__).resolve().parents[1] / "scripts" / "prepare_data.py"
    spec = importlib.util.spec_from_file_location("prepare_data_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def add_regular(archive, name, payload):
    member = tarfile.TarInfo(name)
    member.size = len(payload)
    archive.addfile(member, io.BytesIO(payload))


def test_clip_extraction_flattens_traversal_and_absolute_paths(pipeline, tmp_path):
    archive = tmp_path / "clips.tar.gz"
    destination = tmp_path / "safe"
    escaped = tmp_path / "outside" / "dia9_utt4.mp4"
    with tarfile.open(archive, "w:gz") as output:
        add_regular(output, "../../outside/dia7_utt2.mp4", b"first clip")
        add_regular(output, str(escaped), b"second clip")
        add_regular(output, "../../unrelated.txt", b"must not be extracted")
    names = pipeline.unpack_clips(archive, destination)
    assert names == {"dia7_utt2.mp4", "dia9_utt4.mp4"}
    assert (destination / "dia7_utt2.mp4").read_bytes() == b"first clip"
    assert (destination / "dia9_utt4.mp4").read_bytes() == b"second clip"
    assert set(destination.iterdir()) == {destination / name for name in names}
    assert not escaped.exists()
    assert not (tmp_path / "outside").exists()


@pytest.mark.parametrize("link_type", [tarfile.SYMTYPE, tarfile.LNKTYPE])
@pytest.mark.parametrize("kind", ["clip", "split"])
def test_archive_links_cannot_be_followed(pipeline, tmp_path, link_type, kind):
    archive = tmp_path / "linked.tar.gz"
    name = "dia1_utt0.mp4" if kind == "clip" else "train.tar.gz"
    with tarfile.open(archive, "w:gz") as output:
        member = tarfile.TarInfo(name)
        member.type = link_type
        member.linkname = "/etc/passwd"
        output.addfile(member)
    destination = tmp_path / "safe"
    with pytest.raises(ValueError, match="Invalid"):
        if kind == "clip":
            pipeline.unpack_clips(archive, destination)
        else:
            pipeline.unpack_split_archives(archive, destination, ["train"])
    assert not (destination / name).exists()


def test_duplicate_flattened_clip_names_fail_instead_of_overwriting(pipeline, tmp_path):
    archive = tmp_path / "duplicates.tar.gz"
    with tarfile.open(archive, "w:gz") as output:
        add_regular(output, "first/dia1_utt0.mp4", b"original")
        add_regular(output, "second/dia1_utt0.mp4", b"different")
    with pytest.raises(ValueError, match="duplicate"):
        pipeline.unpack_clips(archive, tmp_path / "safe")
    assert (tmp_path / "safe" / "dia1_utt0.mp4").read_bytes() == b"original"


def test_outer_archive_uses_generated_split_destination(pipeline, tmp_path):
    archive = tmp_path / "outer.tar.gz"
    with tarfile.open(archive, "w:gz") as output:
        add_regular(output, "../../outside/train.tar.gz", b"nested archive bytes")
        add_regular(output, "../../outside/unrequested.tar.gz", b"ignored")
    destination = tmp_path / "safe"
    paths = pipeline.unpack_split_archives(archive, destination, ["train"])
    assert paths == {"train": destination / "train.tar.gz"}
    assert paths["train"].read_bytes() == b"nested archive bytes"
    assert not (tmp_path / "outside").exists()
    assert not (destination / "unrequested.tar.gz").exists()


@pytest.fixture
def record():
    return {
        "id": "test/dia7_utt2",
        "split": "test",
        "dialogue_id": 7,
        "utterance_id": 2,
        "text": "A synthetic utterance for a test.",
        "label": "neutral",
        "speaker": "Synthetic speaker",
    }


@pytest.fixture
def decoder(pipeline, monkeypatch):
    calls = []

    def run(command, **kwargs):
        source = Path(command[command.index("-i") + 1])
        payload = source.read_bytes()
        calls.append(source.name)
        with wave.open(str(command[-1]), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(16000)
            output.writeframes(struct.pack("<h", payload[0]) * 320)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(pipeline.subprocess, "run", run)
    return calls


def test_corrected_test_variant_replaces_cached_plain_source(pipeline, tmp_path, record, decoder):
    video, audio = tmp_path / "video", tmp_path / "audio"
    video.mkdir()
    audio.mkdir()
    plain = video / "dia7_utt2.mp4"
    plain.write_bytes(b"A synthetic plain source")
    before = pipeline.convert_clip(record, video, audio, "fake-ffmpeg")
    assert before["audio_available"]
    assert before["audio_source_filename"] == plain.name

    corrected = video / "final_videos_testdia7_utt2.mp4"
    corrected.write_bytes(b"B synthetic corrected source")
    after = pipeline.convert_clip(record, video, audio, "fake-ffmpeg")
    assert after["audio_available"]
    assert after["audio_source_filename"] == corrected.name
    assert after["audio_source_sha256"] == pipeline.sha256(corrected)
    assert after["audio_sha256"] != before["audio_sha256"]
    assert after["id"] == record["id"] and after["text"] == record["text"]
    assert decoder == [plain.name, corrected.name]

    cached = pipeline.convert_clip(record, video, audio, "fake-ffmpeg")
    assert cached["audio_sha256"] == after["audio_sha256"]
    assert decoder == [plain.name, corrected.name]


def test_same_filename_size_and_timestamp_with_changed_bytes_invalidates_cache(
    pipeline, tmp_path, record, decoder
):
    video, audio = tmp_path / "video", tmp_path / "audio"
    video.mkdir()
    audio.mkdir()
    source = video / "final_videos_testdia7_utt2.mp4"
    source.write_bytes(b"A first version")
    original_stat = source.stat()
    first = pipeline.convert_clip(record, video, audio, "fake-ffmpeg")
    source.write_bytes(b"B other version")
    os.utime(source, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
    assert source.stat().st_size == original_stat.st_size
    second = pipeline.convert_clip(record, video, audio, "fake-ffmpeg")
    assert second["audio_available"]
    assert second["audio_source_sha256"] != first["audio_source_sha256"]
    assert second["audio_sha256"] != first["audio_sha256"]
    assert decoder == [source.name, source.name]


def test_missing_media_retains_annotation_and_explicit_fallback(
    pipeline, tmp_path, record, decoder
):
    audio = tmp_path / "audio"
    audio.mkdir()
    output = pipeline.mark_audio_eligibility(
        pipeline.convert_clip(record, tmp_path / "missing", audio, "fake-ffmpeg")
    )
    assert all(output[key] == value for key, value in record.items())
    assert not output["audio_available"]
    assert not output["audio_usable"]
    assert output["audio_exclusion_reason"] == "missing_or_corrupt_source"
    assert output["audio_error"]
    assert Path(output["audio_path"]).is_absolute()
    assert not Path(output["audio_path"]).exists()
    assert "audio_available" not in record
    assert decoder == []


@pytest.mark.parametrize("duration,eligible", [(29.999, True), (30.0, True), (30.0000625, False)])
def test_model_audio_limit_has_exact_30_second_boundary(pipeline, duration, eligible):
    record = {
        "id": "test/dia2_utt3",
        "audio_available": True,
        "duration_seconds": duration,
    }
    output = pipeline.mark_audio_eligibility(record)
    assert output["audio_available"]  # Exclusion does not claim the file is corrupt.
    assert output["audio_usable"] is eligible
    if not eligible:
        assert output["audio_exclusion_reason"] == "exceeds_30_second_runtime_limit"


def test_corrupt_media_decode_failure_retains_annotation(pipeline, tmp_path, record, monkeypatch):
    video, audio = tmp_path / "video", tmp_path / "audio"
    video.mkdir()
    audio.mkdir()
    (video / "dia7_utt2.mp4").write_bytes(b"corrupt synthetic container")
    monkeypatch.setattr(
        pipeline.subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(
            command, 1, stdout="", stderr="moov atom not found"
        ),
    )
    output = pipeline.mark_audio_eligibility(
        pipeline.convert_clip(record, video, audio, "fake-ffmpeg")
    )
    assert all(output[key] == value for key, value in record.items())
    assert not output["audio_available"] and not output["audio_usable"]
    assert "moov atom not found" in output["audio_error"]
    assert not Path(output["audio_path"]).exists()
