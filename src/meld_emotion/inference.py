"""Local, inspectable inference; no model-generated claims or remote services."""

import json
import time
from pathlib import Path

import numpy as np
from scipy.special import softmax

from .constants import AUDIO_REVISION, LABELS, MAX_AUDIO_SECONDS, TEXT_REVISION
from .encoders import Encoders, prepare_audio


class Heads:
    def __init__(self, model_dir):
        model_dir = Path(model_dir)
        self.metadata = json.loads((model_dir / "metadata.json").read_text())
        if self.metadata["labels"] != LABELS or self.metadata["format_version"] != 1:
            raise ValueError("Incompatible model metadata")
        if (
            self.metadata["text_revision"] != TEXT_REVISION
            or self.metadata["audio_revision"] != AUDIO_REVISION
        ):
            raise ValueError("Encoder revisions do not match the trained heads")
        with np.load(model_dir / "heads.npz", allow_pickle=False) as data:
            self.weights = {k: data[k] for k in data.files}
        dimensions = {}
        for modality in ("text", "audio", "fusion"):
            required = [
                f"{modality}_{key}" for key in ("mean", "scale", "coef", "intercept", "temperature")
            ]
            if any(key not in self.weights for key in required):
                raise ValueError(f"Missing arrays for {modality} head")
            mean, scale, coef, bias, temperature = [self.weights[key] for key in required]
            if any(not np.isfinite(self.weights[key]).all() for key in required):
                raise ValueError(f"Non-finite arrays in {modality} head")
            if mean.ndim != 1 or not mean.size:
                raise ValueError("Head mean must be a nonempty vector")
            d = mean.size
            dimensions[modality] = d
            if scale.shape != (d,) or np.any(scale <= 0):
                raise ValueError("Head scale must be a positive vector")
            if coef.shape != (7, d) or bias.shape != (7,):
                raise ValueError("Head coefficients must have seven class rows")
            if temperature.shape != () or float(temperature) <= 0:
                raise ValueError("Head temperature must be a positive scalar")
        if dimensions["fusion"] != dimensions["text"] + dimensions["audio"]:
            raise ValueError("Fusion dimension must equal text plus audio dimensions")
        threshold = self.metadata.get("uncertainty_threshold")
        if not isinstance(threshold, (float, int)) or not 0 <= threshold <= 1:
            raise ValueError("Uncertainty threshold must be between zero and one")
        mixture = self.weights.get("fusion_mix", np.array(1.0))
        if mixture.shape != () or not np.isfinite(mixture) or not 0 < float(mixture) <= 1:
            raise ValueError("Fusion mixture must be a scalar in (0,1]")
        self.fusion_mix = float(mixture)

    def probabilities(self, features, modality):
        w = self.weights
        scaled = (features - w[f"{modality}_mean"]) / w[f"{modality}_scale"]
        logits = scaled @ w[f"{modality}_coef"].T + w[f"{modality}_intercept"]
        return softmax(logits / w[f"{modality}_temperature"], axis=-1)

    def multimodal_probabilities(self, text_features, audio_features):
        fused = self.probabilities(
            np.concatenate([text_features, audio_features], axis=-1), "fusion"
        )
        text = self.probabilities(text_features, "text")
        return self.fusion_mix * fused + (1 - self.fusion_mix) * text


def grounded_response(text, emotion, uncertain):
    """Quote actual input, then choose a response from the predicted state."""
    words = " ".join(text.split()).split(" ")
    excerpt = " ".join(words[:12]) + ("…" if len(words) > 12 else "")
    if uncertain:
        continuation = "I'm not sure how you're feeling. Would you like to say more?"
    else:
        continuation = {
            "neutral": "I'm listening. What would you like to do next?",
            "joy": "That sounds positive. What are you most excited about?",
            "sadness": "That sounds difficult. Would you like to talk about it?",
            "anger": "That sounds frustrating. What would help right now?",
            "fear": "That sounds worrying. Would it help to talk through the next step?",
            "disgust": "That sounds unpleasant. What would you like to change?",
            "surprise": "That sounds unexpected. How do you feel about it?",
        }[emotion]
    return f"About “{excerpt}”: {continuation}"


class EmotionEngine:
    def __init__(self, model_dir="artifacts/model", device="cpu", local_files_only=True):
        self.heads = Heads(model_dir)
        self.encoders = Encoders(device=device, local_files_only=local_files_only)
        self.parameter_counts = {
            **self.encoders.parameter_counts,
            "heads_normalization_calibration": sum(v.size for v in self.heads.weights.values()),
            "response_policy": 0,
        }
        self.parameter_counts["total"] = sum(self.parameter_counts.values())
        if self.parameter_counts["total"] > 6_000_000_000:
            raise ValueError("Local inference exceeds the 6B total parameter limit")

    def transcribe(self, audio, sample_rate):
        """Recognize speech locally using the existing audio encoder and its CTC head."""
        return self.encoders.transcribe(audio, sample_rate)

    def predict(self, text, audio, sample_rate):
        start = time.perf_counter()
        if not isinstance(text, str) or not text.strip() or len(text) > 4000:
            raise ValueError("Transcript must contain 1–4000 characters.")
        x = None if audio is None else prepare_audio(audio, sample_rate)
        if x is not None and len(x) / 16000 > MAX_AUDIO_SECONDS:
            raise ValueError(f"Audio must be at most {MAX_AUDIO_SECONDS} seconds.")
        x_text = self.encoders.text([text])[0]
        text_prob = self.heads.probabilities(x_text, "text")
        audio_prob = None
        if audio is None:
            status = "text_only_missing_audio"
            probabilities = text_prob
        else:
            if np.std(x) < 1e-5:
                status = "text_only_silent_audio"
                probabilities = text_prob
            else:
                # Validate/resample the original input in the encoder; resampling
                # can overshoot [-1,1], so do not revalidate its intermediate PCM.
                x_audio = self.encoders.audio(audio, sample_rate)
                audio_prob = self.heads.probabilities(x_audio, "audio")
                probabilities = self.heads.multimodal_probabilities(x_text, x_audio)
                status = "text_and_audio"
        emotion = LABELS[int(probabilities.argmax())]
        confidence = float(probabilities.max())
        disagreement = audio_prob is not None and int(audio_prob.argmax()) != int(
            text_prob.argmax()
        )
        uncertain = confidence < self.heads.metadata["uncertainty_threshold"]
        return {
            "emotion": emotion,
            "confidence": confidence,
            "probabilities": dict(zip(LABELS, map(float, probabilities), strict=True)),
            "modality_status": status,
            "modality_disagreement": bool(disagreement),
            "uncertain": uncertain,
            "text_emotion": LABELS[int(text_prob.argmax())],
            "audio_emotion": None if audio_prob is None else LABELS[int(audio_prob.argmax())],
            "response": grounded_response(text, emotion, uncertain),
            "response_policy": "quoted_input_and_predicted_emotion_v1",
            "latency_ms": (time.perf_counter() - start) * 1000,
            "device": self.encoders.device,
        }
