import base64

import numpy as np
import pytest

from meld_emotion.streaming import ProtocolError, StreamSession


def start(session, uid="one", time=0, rate=16_000):
    return session.accept(
        {"type": "start", "utterance_id": uid, "timestamp_ms": time, "sample_rate": rate}
    )


def audio_event(samples=None, uid="one", time=2, sequence=0):
    if samples is None:
        samples = np.zeros(1600, dtype="<f4")
    return {
        "type": "audio",
        "utterance_id": uid,
        "timestamp_ms": time,
        "sequence": sequence,
        "encoding": "pcm_f32le",
        "data": base64.b64encode(np.asarray(samples, dtype="<f4").tobytes()).decode(),
    }


def test_finalization_and_reset_do_not_leak_text_or_audio():
    session = StreamSession()
    start(session)
    session.accept(
        {"type": "text", "utterance_id": "one", "timestamp_ms": 1, "text": "first utterance"}
    )
    session.accept(audio_event(np.full(1600, 0.25)))
    completed = session.accept({"type": "end", "utterance_id": "one", "timestamp_ms": 3})
    assert completed.text == "first utterance"
    assert completed.sample_rate == 16_000
    np.testing.assert_array_equal(completed.audio, np.full(1600, 0.25))
    assert session.sample_count == 0
    assert session.chunks == []
    start(session, "two", time=4)
    session.accept(audio_event(uid="two", time=5))
    completed = session.accept({"type": "end", "utterance_id": "two", "timestamp_ms": 6})
    assert completed.text == ""  # Blank text explicitly requests automatic transcription.
    assert completed.utterance_id == "two"
    session.accept({"type": "reset", "timestamp_ms": 7})
    assert session.text == ""
    assert session.sample_count == 0


def test_sequence_and_timestamp_rejections_do_not_consume_audio():
    session = StreamSession()
    start(session, time=10)
    with pytest.raises(ProtocolError, match="backward"):
        session.accept(audio_event(time=9))
    with pytest.raises(ProtocolError, match="sequence 0"):
        session.accept(audio_event(time=11, sequence=1))
    session.accept(audio_event(time=11))
    with pytest.raises(ProtocolError, match="sequence 1"):
        session.accept(audio_event(time=12))
    assert session.sample_count == 1600
    assert session.next_sequence == 1
    assert session.last_timestamp_ms == 11


@pytest.mark.parametrize("samples", [[np.nan], [np.inf], [-np.inf], [1.01], [-1.01]])
def test_rejects_nonfinite_and_out_of_range_pcm(samples):
    session = StreamSession()
    start(session)
    with pytest.raises(ProtocolError, match="finite"):
        session.accept(audio_event(samples))
    assert session.sample_count == 0


def test_bounds_and_malformed_audio():
    session = StreamSession(max_seconds=0.1, max_chunk_bytes=6400)
    start(session)
    event = audio_event()
    event["data"] = "not-base64!"
    with pytest.raises(ProtocolError, match="base64"):
        session.accept(event)
    event["data"] = base64.b64encode(b"123").decode()
    with pytest.raises(ProtocolError, match="complete float32"):
        session.accept(event)
    with pytest.raises(ProtocolError, match="size limit"):
        session.accept(audio_event(np.zeros(1601)))
    session.accept(audio_event())
    with pytest.raises(ProtocolError, match="utterance limit"):
        session.accept(audio_event([0], sequence=1, time=3))


def test_rejects_wrong_id_and_missing_start():
    session = StreamSession()
    with pytest.raises(ProtocolError, match="send start"):
        session.accept(audio_event())
    start(session)
    with pytest.raises(ProtocolError, match="does not match"):
        session.accept(audio_event(uid="future"))
    with pytest.raises(ProtocolError, match="finish or reset"):
        start(session, "future", time=3)


@pytest.mark.parametrize("timestamp", [True, -1, float("inf"), float("nan"), "1", None])
def test_timestamp_requires_nonnegative_finite_number(timestamp):
    with pytest.raises(ProtocolError, match="timestamp_ms"):
        start(StreamSession(), time=timestamp)


def test_text_updates_replace_instead_of_append():
    session = StreamSession(max_text_chars=5)
    start(session)
    session.accept({"type": "text", "utterance_id": "one", "timestamp_ms": 1, "text": "hi"})
    session.accept({"type": "text", "utterance_id": "one", "timestamp_ms": 2, "text": "bye"})
    with pytest.raises(ProtocolError, match="at most"):
        session.accept(
            {"type": "text", "utterance_id": "one", "timestamp_ms": 3, "text": "toolong"}
        )
    assert session.text == "bye"


def test_audio_minimum_and_connection_clock_survive_reset():
    session = StreamSession()
    start(session)
    session.accept({"type": "text", "utterance_id": "one", "timestamp_ms": 1, "text": "hello"})
    session.accept(audio_event([0]))
    with pytest.raises(ProtocolError, match="100 milliseconds"):
        session.accept({"type": "end", "utterance_id": "one", "timestamp_ms": 3})
    session.accept({"type": "reset", "timestamp_ms": 4})
    with pytest.raises(ProtocolError, match="backward"):
        start(session, time=0)
