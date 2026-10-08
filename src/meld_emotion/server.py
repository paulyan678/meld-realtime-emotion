"""Local-only browser demo and bounded HTTP/WebSocket inference endpoints."""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .streaming import ProtocolError, StreamSession

LOG = logging.getLogger(__name__)
STATIC_DIR = Path(__file__).with_name("static")
MAX_UPLOAD_BYTES = 8 * 1024 * 1024
MAX_SECONDS = 30
MAX_TEXT_CHARS = 4_000
MAX_EVENT_CHARS = 360_000
MAX_CONNECTIONS = 16
IDLE_TIMEOUT_SECONDS = 60


class _RequestTooLarge(Exception):
    pass


async def _run_worker(function: Any, *args: Any) -> Any:
    """Hold our caller's inference lock until its thread actually finishes.

    Cancelling an await does not stop a Python worker thread. Shield both the
    normal wait and cancellation cleanup so a disconnected request cannot let
    the next request enter the still-running model concurrently.
    """
    task = asyncio.create_task(asyncio.to_thread(function, *args))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                continue
            except Exception:
                break
        if not task.cancelled():
            task.exception()  # Retrieve a worker failure during cancellation.
        raise


class _BodyLimit:
    """Bound the actual incoming HTTP body, including chunked requests."""

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        total = 0
        started = False

        async def limited_receive() -> dict:
            nonlocal total
            message = await receive()
            total += len(message.get("body", b""))
            if total > MAX_UPLOAD_BYTES + 65_536:
                raise _RequestTooLarge
            return message

        async def tracked_send(message: dict) -> None:
            nonlocal started
            started |= message["type"] == "http.response.start"
            await send(message)

        try:
            await self.app(scope, limited_receive, tracked_send)
        except _RequestTooLarge:
            if not started:
                response = JSONResponse(
                    {"detail": "request body exceeds 8 MiB plus form overhead"}, status_code=413
                )
                await response(scope, receive, send)


def decode_wav(data: bytes) -> tuple[np.ndarray, int]:
    """Decode a bounded WAV after inspecting metadata, without large allocation."""
    if not data or len(data) > MAX_UPLOAD_BYTES:
        raise ValueError("provide a WAV file of at most 8 MiB")
    try:
        with sf.SoundFile(io.BytesIO(data)) as source:
            if source.format not in {"WAV", "WAVEX"}:
                raise ValueError("audio must be a WAV file")
            if not 8_000 <= source.samplerate <= 48_000:
                raise ValueError("WAV sample rate must be between 8000 and 48000 Hz")
            if not 1 <= source.channels <= 2:
                raise ValueError("WAV must contain one or two channels")
            if not source.samplerate // 10 <= source.frames <= source.samplerate * MAX_SECONDS:
                raise ValueError("WAV duration must be between 0.1 and 30 seconds")
            samples = source.read(dtype="float32", always_2d=True)
            rate = source.samplerate
    except (RuntimeError, sf.LibsndfileError) as exc:
        raise ValueError("cannot decode WAV audio") from exc
    if not np.isfinite(samples).all() or np.any(np.abs(samples) > 1.0):
        raise ValueError("WAV samples must be finite and within [-1, 1]")
    return samples.mean(axis=1, dtype=np.float32), rate


def create_app(engine: Any = None) -> FastAPI:
    """Create an app; model files are loaded only on the first inference request."""
    app = FastAPI(title="MELD · Emotion in context", version="1.0.0")
    app.add_middleware(_BodyLimit)
    app.state.engine = engine
    app.state.inference_lock = asyncio.Lock()
    app.state.sockets = set()
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    async def predict(
        text: str,
        samples: np.ndarray,
        rate: int,
        progress: Callable[[str], Awaitable[None]] | None = None,
    ) -> dict:
        if app.state.inference_lock.locked():
            raise HTTPException(
                429, "inference is busy; retry after the current utterance finishes"
            )
        async with app.state.inference_lock:
            started = time.perf_counter()
            if app.state.engine is None:

                def load_engine() -> Any:
                    from .inference import EmotionEngine

                    return EmotionEngine(
                        model_dir=os.getenv("MELD_MODEL_DIR", "artifacts/model"),
                        device=os.getenv("MELD_DEVICE", "cpu"),
                        local_files_only=os.getenv("MELD_OFFLINE", "1").lower()
                        not in {"0", "false", "no"},
                    )

                try:
                    app.state.engine = await _run_worker(load_engine)
                except Exception as exc:
                    LOG.exception("Cannot load the local emotion model")
                    raise HTTPException(
                        503, "model unavailable; complete model setup and check the server log"
                    ) from exc
            automatic = not text
            transcription_ms = 0.0
            if automatic:
                if progress is not None:
                    await progress("transcribing")
                try:
                    transcription_started = time.perf_counter()
                    text = await _run_worker(app.state.engine.transcribe, samples, rate)
                    transcription_ms = (time.perf_counter() - transcription_started) * 1000
                    if not isinstance(text, str) or not text.strip():
                        raise ValueError(
                            "No speech was recognized. Try again or enter a transcript."
                        )
                    text = text.strip()
                    if len(text) > MAX_TEXT_CHARS:
                        raise ValueError("Recognized transcript exceeds 4000 characters.")
                except ValueError as exc:
                    raise HTTPException(422, str(exc)) from exc
                except OSError as exc:
                    LOG.exception("Cannot load local transcription files")
                    raise HTTPException(
                        503,
                        "transcription model unavailable; run scripts/download_models.py "
                        "or enter a transcript",
                    ) from exc
                except Exception as exc:
                    LOG.exception("Speech transcription failed")
                    raise HTTPException(
                        500, "transcription failed; try again or enter a transcript"
                    ) from exc
                if progress is not None:
                    await progress("analyzing")
            try:
                result = await _run_worker(app.state.engine.predict, text, samples, rate)
                return {
                    **result,
                    "transcript": text,
                    "transcript_source": "automatic" if automatic else "provided",
                    "transcription_ms": transcription_ms,
                    "emotion_latency_ms": result["latency_ms"],
                    "latency_ms": (time.perf_counter() - started) * 1000,
                }
            except ValueError as exc:
                raise HTTPException(422, str(exc)) from exc
            except Exception as exc:
                LOG.exception("Emotion inference failed")
                raise HTTPException(500, "inference failed; check the server log") from exc

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/health")
    async def health() -> dict:
        return {
            "status": "ok",
            "model_loaded": app.state.engine is not None,
            "inference_busy": app.state.inference_lock.locked(),
            "active_streams": len(app.state.sockets),
            "offline": os.getenv("MELD_OFFLINE", "1").lower() not in {"0", "false", "no"},
            "automatic_transcription": True,
            "limits": {"max_audio_seconds": MAX_SECONDS, "max_upload_bytes": MAX_UPLOAD_BYTES},
        }

    @app.post("/predict")
    async def http_predict(request: Request) -> dict:
        try:
            async with request.form(max_files=1, max_fields=1, max_part_size=65_536) as form:
                text = form.get("text", "")
                upload = form.get("audio")
                if not isinstance(text, str) or len(text) > MAX_TEXT_CHARS:
                    raise HTTPException(422, "text must contain at most 4000 characters")
                # Starlette creates UploadFile instances, the superclass of FastAPI's type.
                if upload is None or not hasattr(upload, "read"):
                    raise HTTPException(422, "audio must be an uploaded WAV file")
                data = await upload.read(MAX_UPLOAD_BYTES + 1)
                try:
                    samples, rate = decode_wav(data)
                except ValueError as exc:
                    raise HTTPException(422, str(exc)) from exc
                return await predict(text.strip(), samples, rate)
        except _RequestTooLarge:
            raise
        except HTTPException:
            raise
        except (ValueError, KeyError) as exc:
            raise HTTPException(400, "invalid multipart request") from exc

    @app.websocket("/stream")
    async def stream(socket: WebSocket) -> None:
        if len(app.state.sockets) >= MAX_CONNECTIONS:
            await socket.close(code=1013, reason="too many active streams")
            return
        app.state.sockets.add(socket)
        session = StreamSession(max_seconds=MAX_SECONDS, max_text_chars=MAX_TEXT_CHARS)
        try:
            await socket.accept()
            await socket.send_json(
                {"type": "ready", "max_audio_seconds": MAX_SECONDS, "encoding": "pcm_f32le"}
            )
            while True:
                try:
                    raw = await asyncio.wait_for(
                        socket.receive_text(), timeout=IDLE_TIMEOUT_SECONDS
                    )
                except TimeoutError:
                    await socket.close(code=1001, reason="inactive stream expired")
                    break
                if len(raw) > MAX_EVENT_CHARS:
                    await socket.close(code=1009, reason="event exceeds the message size limit")
                    break
                try:
                    event = json.loads(raw)
                    completed = session.accept(event)
                except (json.JSONDecodeError, ProtocolError, TypeError, KeyError) as exc:
                    await socket.send_json(
                        {"type": "error", "code": "invalid_event", "message": str(exc)}
                    )
                    continue
                if completed is None:
                    await socket.send_json(
                        {
                            "type": "ack",
                            "event": event["type"],
                            "utterance_id": session.utterance_id,
                            "audio_samples": session.sample_count,
                            "next_sequence": session.next_sequence,
                        }
                    )
                    continue
                try:

                    async def progress(stage: str, uid: str = completed.utterance_id) -> None:
                        await socket.send_json(
                            {
                                "type": "processing",
                                "stage": stage,
                                "utterance_id": uid,
                            }
                        )

                    result = await predict(
                        completed.text, completed.audio, completed.sample_rate, progress
                    )
                    await socket.send_json(
                        {"type": "result", "utterance_id": completed.utterance_id, **result}
                    )
                except HTTPException as exc:
                    await socket.send_json(
                        {
                            "type": "error",
                            "code": "inference_busy"
                            if exc.status_code == 429
                            else "inference_failed",
                            "status": exc.status_code,
                            "utterance_id": completed.utterance_id,
                            "message": exc.detail,
                        }
                    )
        except WebSocketDisconnect:
            pass
        except (KeyError, RuntimeError):
            # A non-text frame or transport closure must release its bounded slot.
            LOG.debug("WebSocket transport closed", exc_info=True)
        finally:
            session.clear()
            app.state.sockets.discard(socket)

    return app


app = create_app()
