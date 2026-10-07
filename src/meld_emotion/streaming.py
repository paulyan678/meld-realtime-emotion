"""Bounded utterance assembly for the public, timestamped WebSocket protocol.

Audio events contain base64 little-endian float32, mono PCM. ``timestamp_ms``
is the client's event clock, not an audio frame offset; it must never decrease.
Only a finalized utterance reaches the model. No state crosses utterances.
"""

from __future__ import annotations

import base64
import binascii
import math
from dataclasses import dataclass
from typing import Any

import numpy as np


class ProtocolError(ValueError):
    """A client event was invalid and was not applied."""


@dataclass(frozen=True)
class CompletedUtterance:
    utterance_id: str
    text: str
    audio: np.ndarray
    sample_rate: int
    started_at_ms: float
    ended_at_ms: float


class StreamSession:
    """Per-connection state machine, independent of networking and inference."""

    def __init__(
        self,
        *,
        max_seconds: float = 30,
        max_chunk_bytes: int = 262_144,
        max_text_chars: int = 4_000,
    ) -> None:
        self.max_seconds = max_seconds
        self.max_chunk_bytes = max_chunk_bytes
        self.max_text_chars = max_text_chars
        self.last_timestamp_ms = -math.inf
        self.clear()

    def clear(self) -> None:
        """Release the current utterance, retaining the connection event clock."""
        self.utterance_id: str | None = None
        self.sample_rate = 0
        self.started_at_ms = 0.0
        self.text = ""
        self.chunks: list[np.ndarray] = []
        self.sample_count = 0
        self.next_sequence = 0

    @staticmethod
    def _number(value: Any, field: str) -> float:
        if isinstance(value, bool) or not isinstance(value, (float, int)):
            raise ProtocolError(f"{field} must be a finite number")
        try:
            finite = math.isfinite(value)
        except OverflowError:
            finite = False
        if not finite or value < 0:
            raise ProtocolError(f"{field} must be a finite nonnegative number")
        return float(value)

    def accept(self, event: dict[str, Any]) -> CompletedUtterance | None:
        """Validate and apply one event atomically; end consumes its utterance."""
        if not isinstance(event, dict):
            raise ProtocolError("event must be a JSON object")
        kind = event.get("type")
        if not isinstance(kind, str) or kind not in {"start", "text", "audio", "end", "reset"}:
            raise ProtocolError("unknown event type")
        timestamp = self._number(event.get("timestamp_ms"), "timestamp_ms")
        if timestamp < self.last_timestamp_ms:
            raise ProtocolError("timestamp_ms must not move backward")

        if kind == "reset":
            self.clear()
            self.last_timestamp_ms = timestamp
            return None

        utterance_id = event.get("utterance_id")
        if not isinstance(utterance_id, str) or not 1 <= len(utterance_id) <= 128:
            raise ProtocolError("utterance_id must be a string of 1–128 characters")
        if kind == "start":
            if self.utterance_id is not None:
                raise ProtocolError("finish or reset the current utterance before start")
            rate = event.get("sample_rate")
            if isinstance(rate, bool) or not isinstance(rate, int) or not 8_000 <= rate <= 48_000:
                raise ProtocolError("sample_rate must be an integer from 8000 to 48000")
            self.utterance_id = utterance_id
            self.sample_rate = rate
            self.started_at_ms = timestamp
        else:
            if self.utterance_id is None:
                raise ProtocolError("send start before utterance events")
            if utterance_id != self.utterance_id:
                raise ProtocolError("utterance_id does not match the active utterance")
            if kind == "text":
                text = event.get("text")
                if not isinstance(text, str) or len(text) > self.max_text_chars:
                    raise ProtocolError(
                        f"text must contain at most {self.max_text_chars} characters"
                    )
                self.text = text.strip()
            elif kind == "audio":
                self._accept_audio(event)
            elif kind == "end":
                if self.sample_count < self.sample_rate // 10:
                    raise ProtocolError("at least 100 milliseconds of audio are required")
                completed = CompletedUtterance(
                    utterance_id=self.utterance_id,
                    text=self.text,
                    audio=np.concatenate(self.chunks),
                    sample_rate=self.sample_rate,
                    started_at_ms=self.started_at_ms,
                    ended_at_ms=timestamp,
                )
                self.clear()
                self.last_timestamp_ms = timestamp
                return completed
        self.last_timestamp_ms = timestamp
        return None

    def _accept_audio(self, event: dict[str, Any]) -> None:
        sequence = event.get("sequence")
        if (
            isinstance(sequence, bool)
            or not isinstance(sequence, int)
            or sequence != self.next_sequence
        ):
            raise ProtocolError(f"expected audio sequence {self.next_sequence}")
        if event.get("encoding") != "pcm_f32le":
            raise ProtocolError("audio encoding must be pcm_f32le")
        data = event.get("data")
        if not isinstance(data, str) or len(data) > 4 * ((self.max_chunk_bytes + 2) // 3):
            raise ProtocolError("audio chunk exceeds the size limit")
        try:
            raw = base64.b64decode(data, validate=True)
        except (ValueError, binascii.Error) as exc:
            raise ProtocolError("audio data must be valid base64") from exc
        if not raw or len(raw) % 4 or len(raw) > self.max_chunk_bytes:
            raise ProtocolError(
                "audio chunk must contain complete float32 samples within the size limit"
            )
        samples = np.frombuffer(raw, dtype="<f4")
        if not np.isfinite(samples).all() or np.any(np.abs(samples) > 1.0):
            raise ProtocolError("PCM samples must be finite and within [-1, 1]")
        if self.sample_count + len(samples) > int(self.sample_rate * self.max_seconds):
            raise ProtocolError(f"audio exceeds the {self.max_seconds:g}-second utterance limit")
        self.chunks.append(samples)
        self.sample_count += len(samples)
        self.next_sequence += 1
