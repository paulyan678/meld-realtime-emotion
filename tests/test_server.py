import asyncio
import base64
import io
import threading
from concurrent.futures import ThreadPoolExecutor

import httpx
import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from meld_emotion import server
from meld_emotion.server import create_app, decode_wav


class FakeEngine:
    def __init__(self):
        self.calls = []
        self.failure = None
        self.transcription_calls = []
        self.transcription_failure = None

    def transcribe(self, audio, sample_rate):
        self.transcription_calls.append((audio.copy(), sample_rate))
        if self.transcription_failure:
            raise self.transcription_failure
        return "I am glad to hear that"

    def predict(self, text, audio, sample_rate):
        self.calls.append((text, audio.copy(), sample_rate))
        if self.failure:
            raise self.failure
        return {
            "emotion": "joy",
            "confidence": 0.8,
            "probabilities": {"joy": 0.8, "neutral": 0.2},
            "response": f'I heard: "{text}".',
            "modality_status": {"text": "available", "audio": "available"},
            "latency_ms": 12.5,
        }


def wav_bytes(samples=None, rate=16_000, subtype="PCM_16"):
    target = io.BytesIO()
    sf.write(
        target, np.zeros(3200) if samples is None else samples, rate, format="WAV", subtype=subtype
    )
    return target.getvalue()


def send_utterance(ws, uid="one", time=0, text="Hello there", value=0.2):
    ws.send_json(
        {"type": "start", "utterance_id": uid, "timestamp_ms": time, "sample_rate": 16_000}
    )
    assert ws.receive_json()["type"] == "ack"
    ws.send_json({"type": "text", "utterance_id": uid, "timestamp_ms": time + 1, "text": text})
    assert ws.receive_json()["type"] == "ack"
    ws.send_json(
        {
            "type": "audio",
            "utterance_id": uid,
            "timestamp_ms": time + 2,
            "sequence": 0,
            "encoding": "pcm_f32le",
            "data": base64.b64encode(np.full(1600, value, dtype="<f4").tobytes()).decode(),
        }
    )
    assert ws.receive_json()["audio_samples"] == 1600
    ws.send_json({"type": "end", "utterance_id": uid, "timestamp_ms": time + 3})
    return ws.receive_json()


def test_home_health_and_static_assets_work_without_loading_models():
    with TestClient(create_app()) as client:
        health = client.get("/health").json()
        assert health["status"] == "ok"
        assert health["model_loaded"] is False
        assert health["offline"] is True
        assert "Speech emotion demo" in client.get("/").text
        assert client.get("/static/app.js").status_code == 200


def test_http_prediction_decodes_stereo_and_preserves_transcript():
    engine = FakeEngine()
    with TestClient(create_app(engine)) as client:
        response = client.post(
            "/predict",
            data={"text": "  It's a good day.  "},
            files={"audio": ("test.wav", wav_bytes(np.full((3200, 2), 0.25)), "audio/wav")},
        )
        assert response.status_code == 200
        assert response.json()["emotion"] == "joy"
        assert response.json()["transcript_source"] == "provided"
        assert response.json()["transcript"] == "It's a good day."
        assert response.json()["transcription_ms"] == 0
        assert engine.transcription_calls == []
    text, samples, rate = engine.calls[0]
    assert text == "It's a good day."
    assert samples.shape == (3200,)
    assert samples.dtype == np.float32
    assert rate == 16_000
    np.testing.assert_allclose(samples, 0.25)


@pytest.mark.parametrize(
    "payload,text",
    [
        (b"garbage", "hello"),
        (wav_bytes(), "x" * 4001),
        (wav_bytes(np.zeros(10)), "hello"),
    ],
)
def test_http_rejects_invalid_audio_and_text_without_inference(payload, text):
    engine = FakeEngine()
    with TestClient(create_app(engine)) as client:
        response = client.post(
            "/predict", data={"text": text}, files={"audio": ("test.wav", payload)}
        )
        assert response.status_code == 422
    assert engine.calls == []


def test_http_rejects_oversized_actual_body():
    with TestClient(create_app(FakeEngine())) as client:
        response = client.post(
            "/predict",
            data={"text": "hello"},
            files={"audio": ("test.wav", b"0" * (server.MAX_UPLOAD_BYTES + 70_000))},
        )
        assert response.status_code == 413


@pytest.mark.parametrize(
    "samples,rate",
    [
        (np.zeros(480_001), 16_000),
        (np.zeros(3200), 96_000),
        (np.zeros((3200, 3)), 16_000),
        (np.full(3200, np.nan), 16_000),
        (np.full(3200, 1.1), 16_000),
    ],
)
def test_wav_metadata_duration_and_sample_validation(samples, rate):
    with pytest.raises(ValueError):
        decode_wav(wav_bytes(samples, rate, subtype="FLOAT"))


def test_stream_results_are_isolated_between_utterances():
    engine = FakeEngine()
    with TestClient(create_app(engine)) as client:
        with client.websocket_connect("/stream") as ws:
            assert ws.receive_json()["type"] == "ready"
            first = send_utterance(ws)
            second = send_utterance(ws, uid="two", time=4, text="Second input", value=-0.5)
            assert first["utterance_id"] == "one"
            assert second["utterance_id"] == "two"
            assert client.get("/health").json()["active_streams"] == 1
        assert client.get("/health").json()["active_streams"] == 0
    assert [call[0] for call in engine.calls] == ["Hello there", "Second input"]
    assert all(len(call[1]) == 1600 for call in engine.calls)
    np.testing.assert_allclose(engine.calls[1][1], -0.5)


def test_protocol_error_is_recoverable_and_does_not_call_model():
    engine = FakeEngine()
    with TestClient(create_app(engine)) as client:
        with client.websocket_connect("/stream") as ws:
            ws.receive_json()
            ws.send_text("not json")
            assert ws.receive_json()["code"] == "invalid_event"
            ws.send_json({"type": "end", "utterance_id": "one", "timestamp_ms": 0})
            assert "send start" in ws.receive_json()["message"]
            assert engine.calls == []
            assert send_utterance(ws)["type"] == "result"


def test_inference_failure_consumes_utterance_and_does_not_leak_exception():
    engine = FakeEngine()
    engine.failure = RuntimeError("secret internal path")
    with TestClient(create_app(engine)) as client:
        with client.websocket_connect("/stream") as ws:
            ws.receive_json()
            result = send_utterance(ws)
            assert result["status"] == 500
            assert "secret" not in result["message"]
            ws.send_json({"type": "end", "utterance_id": "one", "timestamp_ms": 4})
            assert "send start" in ws.receive_json()["message"]
            engine.failure = None
            assert send_utterance(ws, uid="next", time=5)["type"] == "result"


def test_busy_inference_is_rejected_without_queueing():
    entered = threading.Event()
    release = threading.Event()

    class BlockingEngine(FakeEngine):
        def predict(self, *args):
            entered.set()
            assert release.wait(5)
            return super().predict(*args)

    engine = BlockingEngine()
    with TestClient(create_app(engine)) as client, ThreadPoolExecutor(max_workers=1) as pool:
        first = pool.submit(
            client.post, "/predict", data={"text": "one"}, files={"audio": ("one.wav", wav_bytes())}
        )
        assert entered.wait(3)
        try:
            second = client.post(
                "/predict", data={"text": "two"}, files={"audio": ("two.wav", wav_bytes())}
            )
            assert second.status_code == 429
            assert client.get("/health").json()["inference_busy"] is True
        finally:
            release.set()
        assert first.result(timeout=3).status_code == 200
        assert len(engine.calls) == 1


def test_socket_count_is_bounded_and_slots_release(monkeypatch):
    monkeypatch.setattr(server, "MAX_CONNECTIONS", 1)
    with TestClient(create_app(FakeEngine())) as client:
        with client.websocket_connect("/stream") as first:
            first.receive_json()
            with pytest.raises(WebSocketDisconnect) as exc:
                with client.websocket_connect("/stream"):
                    pass
            assert exc.value.code == 1013
        with client.websocket_connect("/stream") as replacement:
            assert replacement.receive_json()["type"] == "ready"


def test_request_cancellation_holds_lock_until_worker_completes():
    entered = threading.Event()
    release = threading.Event()

    class BlockingEngine(FakeEngine):
        def predict(self, *args):
            entered.set()
            assert release.wait(5)
            return super().predict(*args)

    async def scenario():
        app = create_app(BlockingEngine())
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            first = asyncio.create_task(
                client.post(
                    "/predict", data={"text": "one"}, files={"audio": ("one.wav", wav_bytes())}
                )
            )
            assert await asyncio.to_thread(entered.wait, 3)
            first.cancel()
            await asyncio.sleep(0)
            try:
                health = await client.get("/health")
                assert health.json()["inference_busy"] is True
                second = await client.post(
                    "/predict", data={"text": "two"}, files={"audio": ("two.wav", wav_bytes())}
                )
                assert second.status_code == 429
            finally:
                release.set()
            with pytest.raises(asyncio.CancelledError):
                await first
            assert app.state.inference_lock.locked() is False
            assert len(app.state.engine.calls) == 1

    asyncio.run(scenario())


def test_idle_sockets_expire_and_oversized_events_close(monkeypatch):
    monkeypatch.setattr(server, "IDLE_TIMEOUT_SECONDS", 0.05)
    with TestClient(create_app(FakeEngine())) as client:
        with client.websocket_connect("/stream") as ws:
            ws.receive_json()
            with pytest.raises(WebSocketDisconnect) as exc:
                ws.receive_json()
            assert exc.value.code == 1001
        with client.websocket_connect("/stream") as ws:
            ws.receive_json()
            ws.send_text("x" * (server.MAX_EVENT_CHARS + 1))
            with pytest.raises(WebSocketDisconnect) as exc:
                ws.receive_json()
            assert exc.value.code == 1009


@pytest.mark.parametrize("form", [{}, {"text": ""}, {"text": " \n "}])
def test_http_transcribes_when_text_is_omitted_or_blank(form):
    engine = FakeEngine()
    samples = np.sin(np.arange(3200) * 0.1).astype(np.float32) * 0.2
    with TestClient(create_app(engine)) as client:
        response = client.post(
            "/predict", data=form, files={"audio": ("test.wav", wav_bytes(samples))}
        )
    assert response.status_code == 200
    result = response.json()
    assert result["transcript_source"] == "automatic"
    assert result["transcript"] == "I am glad to hear that"
    assert result["transcription_ms"] >= 0
    assert result["latency_ms"] >= result["transcription_ms"]
    assert result["emotion_latency_ms"] == 12.5
    assert len(engine.transcription_calls) == 1
    np.testing.assert_array_equal(engine.transcription_calls[0][0], engine.calls[0][1])
    assert engine.calls[0][0] == result["transcript"]
    assert result["transcript"] in result["response"]


@pytest.mark.parametrize(
    "failure,status,message",
    [
        (ValueError("No speech detected"), 422, "No speech detected"),
        (OSError("private cache path"), 503, "transcription model unavailable"),
        (RuntimeError("private internals"), 500, "transcription failed"),
    ],
)
def test_asr_failure_prevents_emotion_inference_and_releases_lock(failure, status, message):
    engine = FakeEngine()
    engine.transcription_failure = failure
    with TestClient(create_app(engine)) as client:
        response = client.post("/predict", files={"audio": ("test.wav", wav_bytes())})
        assert response.status_code == status
        assert message in response.json()["detail"]
        assert "private" not in response.text
        assert engine.calls == []
        assert client.get("/health").json()["inference_busy"] is False
        # A manual correction remains usable even if local ASR is unavailable.
        response = client.post(
            "/predict", data={"text": "Corrected"}, files={"audio": ("test.wav", wav_bytes())}
        )
        assert response.status_code == 200


def test_stream_automatic_transcripts_do_not_leak_into_next_utterance():
    engine = FakeEngine()
    with TestClient(create_app(engine)) as client:
        with client.websocket_connect("/stream") as ws:
            ws.receive_json()
            assert send_utterance(ws, text="")["stage"] == "transcribing"
            assert ws.receive_json()["stage"] == "analyzing"
            result = ws.receive_json()
            assert result["type"] == "result"
            assert result["transcript_source"] == "automatic"
            assert result["transcript"] == "I am glad to hear that"
            second = send_utterance(ws, uid="two", time=4, text="Second manual input")
            assert second["type"] == "result"
            assert second["transcript_source"] == "provided"
            assert second["transcript"] == "Second manual input"
            assert len(engine.transcription_calls) == 1


def test_stream_transcription_error_consumes_turn_and_allows_retry():
    engine = FakeEngine()
    engine.transcription_failure = ValueError("No speech detected")
    with TestClient(create_app(engine)) as client:
        with client.websocket_connect("/stream") as ws:
            ws.receive_json()
            assert send_utterance(ws, text="")["stage"] == "transcribing"
            assert ws.receive_json()["status"] == 422
            assert engine.calls == []
            engine.transcription_failure = None
            assert send_utterance(ws, uid="retry", time=4, text="")["stage"] == "transcribing"
            assert ws.receive_json()["stage"] == "analyzing"
            assert ws.receive_json()["type"] == "result"


def test_transcription_cancellation_keeps_worker_inside_inference_lock():
    entered, release = threading.Event(), threading.Event()

    class BlockingTranscriber(FakeEngine):
        def transcribe(self, *args):
            entered.set()
            assert release.wait(5)
            return super().transcribe(*args)

    async def scenario():
        app = create_app(BlockingTranscriber())
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            first = asyncio.create_task(
                client.post("/predict", files={"audio": ("test.wav", wav_bytes())})
            )
            assert await asyncio.to_thread(entered.wait, 3)
            first.cancel()
            await asyncio.sleep(0)
            try:
                second = await client.post(
                    "/predict",
                    data={"text": "manual"},
                    files={"audio": ("test.wav", wav_bytes())},
                )
                assert second.status_code == 429
            finally:
                release.set()
            with pytest.raises(asyncio.CancelledError):
                await first
            assert app.state.inference_lock.locked() is False
            assert app.state.engine.calls == []

    asyncio.run(scenario())
