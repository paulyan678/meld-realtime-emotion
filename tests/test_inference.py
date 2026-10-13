"""Inference contract tests run without downloading either pretrained encoder."""

import json

import numpy as np
import pytest
from scipy.special import softmax

from meld_emotion import inference
from meld_emotion.constants import AUDIO_REVISION, LABELS, MAX_AUDIO_SECONDS, TEXT_REVISION
from meld_emotion.encoders import prepare_audio


class FakeEncoders:
    parameter_counts = {"text_encoder": 23, "audio_encoder": 41, "asr_ctc_head": 5}

    def __init__(self, device="cpu", local_files_only=True):
        self.device = device
        self.local_files_only = local_files_only
        self.text_calls = []
        self.audio_calls = []

    def text(self, texts):
        self.text_calls.append(texts)
        return np.tile([1.0, 2.0], (len(texts), 1))

    def audio(self, audio, sample_rate):
        self.audio_calls.append((audio, sample_rate))
        return np.array([3.0, 4.0, 5.0])


@pytest.fixture
def model_dir(tmp_path):
    metadata = {
        "format_version": 1,
        "labels": LABELS,
        "text_revision": TEXT_REVISION,
        "audio_revision": AUDIO_REVISION,
        "uncertainty_threshold": 0.45,
    }
    weights = {}
    for modality, dimension, winner in (
        ("text", 2, "joy"),
        ("audio", 3, "anger"),
        ("fusion", 5, "surprise"),
    ):
        bias = np.zeros(len(LABELS))
        bias[LABELS.index(winner)] = 5.0
        weights.update(
            {
                f"{modality}_mean": np.zeros(dimension),
                f"{modality}_scale": np.ones(dimension),
                f"{modality}_coef": np.zeros((len(LABELS), dimension)),
                f"{modality}_intercept": bias,
                f"{modality}_temperature": np.array(1.0),
            }
        )
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))
    np.savez(tmp_path / "heads.npz", **weights)
    return tmp_path


@pytest.fixture
def engine(monkeypatch, model_dir):
    monkeypatch.setattr(inference, "Encoders", FakeEncoders)
    return inference.EmotionEngine(model_dir)


def test_heads_normalize_features_and_apply_temperature(model_dir):
    heads = inference.Heads(model_dir)
    heads.weights["text_mean"] = np.array([2.0, 5.0])
    heads.weights["text_scale"] = np.array([2.0, 4.0])
    heads.weights["text_coef"] = np.column_stack([np.arange(7), np.zeros(7)])
    heads.weights["text_intercept"] = np.zeros(7)
    heads.weights["text_temperature"] = np.array(2.0)
    # First standardized coordinate is 2, so calibrated logits are 0,...,6.
    np.testing.assert_allclose(
        heads.probabilities(np.array([6.0, 5.0]), "text"), softmax(np.arange(7)), rtol=1e-12
    )


def update_weights(model_dir, **updates):
    with np.load(model_dir / "heads.npz", allow_pickle=False) as original:
        weights = {name: original[name] for name in original.files}
    weights.update(updates)
    np.savez(model_dir / "heads.npz", **weights)


@pytest.mark.parametrize("alpha", [0.25, 0.5, 1.0])
@pytest.mark.parametrize("batch_size", [None, 3])
def test_mixture_blends_calibrated_probabilities_for_single_and_batch_inputs(
    model_dir, alpha, batch_size
):
    update_weights(
        model_dir,
        fusion_mix=np.array(alpha),
        text_temperature=np.array(2.0),
        fusion_temperature=np.array(4.0),
    )
    heads = inference.Heads(model_dir)
    text_logits, fusion_logits = np.zeros(7), np.zeros(7)
    text_logits[LABELS.index("joy")] = 2.5  # Original bias 5 / temperature 2.
    fusion_logits[LABELS.index("surprise")] = 1.25  # Original bias 5 / temperature 4.
    expected = alpha * softmax(fusion_logits) + (1 - alpha) * softmax(text_logits)
    text, audio = np.array([1.0, 2.0]), np.array([3.0, 4.0, 5.0])
    if batch_size:
        text, audio = np.tile(text, (batch_size, 1)), np.tile(audio, (batch_size, 1))
        expected = np.tile(expected, (batch_size, 1))
    actual = heads.multimodal_probabilities(text, audio)
    np.testing.assert_allclose(actual, expected, rtol=1e-12)
    np.testing.assert_allclose(actual.sum(axis=-1), 1.0)


@pytest.mark.parametrize("alpha", [0.0, -0.1, 1.1, np.nan, np.inf, np.array([0.5])])
def test_heads_reject_invalid_mixture_weights(model_dir, alpha):
    update_weights(model_dir, fusion_mix=np.asarray(alpha))
    with pytest.raises(ValueError, match="mixture"):
        inference.Heads(model_dir)


def test_engine_uses_saved_mixture_and_counts_its_parameter(monkeypatch, model_dir):
    monkeypatch.setattr(inference, "Encoders", FakeEncoders)
    original_count = inference.EmotionEngine(model_dir).parameter_counts["total"]
    update_weights(model_dir, fusion_mix=np.array(0.25))
    engine = inference.EmotionEngine(model_dir)
    waveform = np.sin(np.arange(800) * 2 * np.pi * 440 / 16000).astype(np.float32) * 0.1
    result = engine.predict("The meeting was moved again.", waveform, 16000)
    expected = engine.heads.multimodal_probabilities(np.array([1, 2]), np.array([3, 4, 5]))
    np.testing.assert_allclose(list(result["probabilities"].values()), expected)
    assert result["emotion"] == "joy"  # Raw concatenation head alone favors surprise.
    assert result["modality_status"] == "text_and_audio"
    assert len(engine.encoders.audio_calls) == 1
    assert engine.parameter_counts["total"] == original_count + 1


@pytest.mark.parametrize("audio", [None, np.zeros(800)])
def test_fallback_remains_pure_text_when_mixture_is_enabled(monkeypatch, model_dir, audio):
    update_weights(model_dir, fusion_mix=np.array(0.25))
    monkeypatch.setattr(inference, "Encoders", FakeEncoders)
    engine = inference.EmotionEngine(model_dir)
    result = engine.predict("The meeting was moved again.", audio, 16000)
    expected = engine.heads.probabilities(np.array([1, 2]), "text")
    np.testing.assert_allclose(list(result["probabilities"].values()), expected)
    assert engine.encoders.audio_calls == []


@pytest.mark.parametrize(
    "field,value",
    [
        ("labels", list(reversed(LABELS))),
        ("format_version", 99),
        ("text_revision", "different-text-weights"),
        ("audio_revision", "different-audio-weights"),
    ],
)
def test_heads_reject_incompatible_model_metadata(model_dir, field, value):
    metadata = json.loads((model_dir / "metadata.json").read_text())
    metadata[field] = value
    (model_dir / "metadata.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError):
        inference.Heads(model_dir)


@pytest.mark.parametrize(
    "key,value",
    [
        ("text_scale", np.zeros(2)),
        ("text_scale", np.array([-1.0, 1.0])),
        ("text_coef", np.full((7, 2), np.nan)),
        ("text_coef", np.zeros((6, 2))),
        ("text_mean", np.zeros(3)),
        ("text_intercept", np.zeros(6)),
        ("text_temperature", np.array(0.0)),
        ("text_temperature", np.array(-1.0)),
        ("text_temperature", np.array(np.inf)),
        ("text_temperature", np.ones(7)),
    ],
)
def test_heads_reject_malformed_weights_on_load(model_dir, key, value):
    with np.load(model_dir / "heads.npz", allow_pickle=False) as original:
        weights = {name: original[name] for name in original.files}
    weights[key] = value
    np.savez(model_dir / "heads.npz", **weights)
    with pytest.raises(ValueError):
        inference.Heads(model_dir)


def test_heads_reject_missing_required_array(model_dir):
    with np.load(model_dir / "heads.npz", allow_pickle=False) as original:
        weights = {name: original[name] for name in original.files if name != "audio_mean"}
    np.savez(model_dir / "heads.npz", **weights)
    with pytest.raises(ValueError):
        inference.Heads(model_dir)


@pytest.mark.parametrize("threshold", [-0.1, 1.1, float("nan")])
def test_heads_reject_invalid_uncertainty_threshold(model_dir, threshold):
    metadata = json.loads((model_dir / "metadata.json").read_text())
    metadata["uncertainty_threshold"] = threshold
    (model_dir / "metadata.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError):
        inference.Heads(model_dir)


def test_fusion_maps_all_probability_columns_to_declared_labels(engine):
    waveform = np.sin(np.arange(800) * 2 * np.pi * 440 / 16000).astype(np.float32) * 0.1
    result = engine.predict("The meeting was moved again.", waveform, 16000)
    assert list(result["probabilities"]) == LABELS
    assert sum(result["probabilities"].values()) == pytest.approx(1.0)
    assert result["confidence"] == result["probabilities"]["surprise"]
    assert result["emotion"] == "surprise"
    assert result["text_emotion"] == "joy"
    assert result["audio_emotion"] == "anger"
    assert result["modality_disagreement"] is True
    assert result["modality_status"] == "text_and_audio"
    assert result["uncertain"] is False
    assert "The meeting was moved again." in result["response"]
    assert result["latency_ms"] >= 0


@pytest.mark.parametrize(
    "audio,status",
    [
        (None, "text_only_missing_audio"),
        (np.zeros(800), "text_only_silent_audio"),
        (np.full(800, 1e-7), "text_only_silent_audio"),
        (np.full(800, 0.1), "text_only_silent_audio"),
    ],
)
def test_absent_or_silent_audio_uses_text_head_without_audio_encoder(engine, audio, status):
    result = engine.predict("I passed the exam.", audio, 16000)
    assert result["emotion"] == "joy"
    assert result["modality_status"] == status
    assert result["audio_emotion"] is None
    assert result["modality_disagreement"] is False
    assert engine.encoders.audio_calls == []


def test_low_confidence_requests_clarification_and_keeps_input_grounding(engine):
    engine.heads.weights["text_intercept"][:] = 0
    result = engine.predict("I got the message.", None, 16000)
    assert result["uncertain"] is True
    assert result["confidence"] == pytest.approx(1 / 7)
    assert "I got the message." in result["response"]
    assert "I'm not sure how you're feeling" in result["response"]


def test_prediction_depends_only_on_current_utterance(engine):
    first = engine.predict("I am reading a book.", None, 16000)
    engine.predict("An unrelated future utterance.", None, 16000)
    repeated = engine.predict("I am reading a book.", None, 16000)
    assert first["probabilities"] == repeated["probabilities"]
    assert first["response"] == repeated["response"]
    assert engine.encoders.text_calls == [
        ["I am reading a book."],
        ["An unrelated future utterance."],
        ["I am reading a book."],
    ]


@pytest.mark.parametrize("text", [None, 12, "", " \n\t", "x" * 4001])
def test_invalid_transcripts_rejected_before_encoding(engine, text):
    with pytest.raises(ValueError, match="Transcript"):
        engine.predict(text, None, 16000)
    assert engine.encoders.text_calls == []


@pytest.mark.parametrize(
    "audio,rate",
    [
        (np.zeros(4), 0),
        (np.zeros(4), -16000),
        (np.zeros(4), 16000.0),
        (np.zeros(4), 7999),
        (np.zeros(4), 96001),
        (np.array(0.1), 16000),
        (np.empty(0), 16000),
        (np.zeros((4, 2)), 16000),
        (np.array([np.nan]), 16000),
        (np.array([np.inf]), 16000),
        (np.array([1.5]), 16000),
    ],
)
def test_invalid_audio_rejected_with_value_error(engine, audio, rate):
    with pytest.raises(ValueError):
        engine.predict("hello", audio, rate)
    assert engine.encoders.audio_calls == []


def test_duration_limit_rejected(engine):
    with pytest.raises(ValueError, match="seconds"):
        engine.predict("hello", np.zeros(16000 * MAX_AUDIO_SECONDS + 1), 16000)
    assert engine.encoders.audio_calls == []


def test_audio_resampling_preserves_duration_and_finite_float_samples():
    original = np.sin(np.arange(8000) * 2 * np.pi * 440 / 8000).astype(np.float32) * 0.2
    resampled = prepare_audio(original, 8000)
    assert resampled.shape == (16000,)
    assert resampled.dtype == np.float32
    assert np.isfinite(resampled).all()
    assert np.sqrt(np.mean(resampled**2)) == pytest.approx(0.2 / np.sqrt(2), rel=0.01)


def test_valid_pcm_is_not_rejected_after_resampling_overshoot(monkeypatch, model_dir):
    class ValidatingEncoders(FakeEncoders):
        def audio(self, audio, sample_rate):
            prepare_audio(audio, sample_rate)
            return super().audio(audio, sample_rate)

    monkeypatch.setattr(inference, "Encoders", ValidatingEncoders)
    engine = inference.EmotionEngine(model_dir)
    # Polyphase interpolation of a valid full-scale signal can exceed +/-1.
    audio = np.tile(np.array([1, 1, -1, -1], dtype=np.float32), 100)
    assert np.max(np.abs(prepare_audio(audio, 8000))) > 1.01
    result = engine.predict("The original recording is valid PCM.", audio, 8000)
    assert result["modality_status"] == "text_and_audio"


def test_all_loaded_model_and_head_values_are_counted(engine):
    head_values = sum(array.size for array in engine.heads.weights.values())
    assert engine.parameter_counts == {
        "text_encoder": 23,
        "audio_encoder": 41,
        "asr_ctc_head": 5,
        "heads_normalization_calibration": head_values,
        "response_policy": 0,
        "total": 23 + 41 + 5 + head_values,
    }
    assert engine.encoders.local_files_only is True


def test_cap_includes_head_parameters_not_only_backbones(monkeypatch, model_dir):
    class OversizedEncoders(FakeEncoders):
        parameter_counts = {"text_encoder": 3_000_000_000, "audio_encoder": 3_000_000_000}

    monkeypatch.setattr(inference, "Encoders", OversizedEncoders)
    with pytest.raises(ValueError, match="6B"):
        inference.EmotionEngine(model_dir)


def test_transcription_head_can_push_complete_path_over_parameter_limit(monkeypatch, model_dir):
    head_values = sum(array.size for array in inference.Heads(model_dir).weights.values())

    class AtLimitEncoders(FakeEncoders):
        parameter_counts = {
            "text_encoder": 3_000_000_000,
            "audio_encoder": 3_000_000_000 - head_values,
            "asr_ctc_head": 0,
        }

    monkeypatch.setattr(inference, "Encoders", AtLimitEncoders)
    assert inference.EmotionEngine(model_dir).parameter_counts["total"] == 6_000_000_000
    monkeypatch.setitem(AtLimitEncoders.parameter_counts, "asr_ctc_head", 1)
    with pytest.raises(ValueError, match="6B"):
        inference.EmotionEngine(model_dir)


def test_response_excerpt_is_whitespace_normalized_and_bounded():
    response = inference.grounded_response(
        "one  two\nthree four five six seven eight nine ten eleven twelve thirteen", "joy", False
    )
    assert "one two three four five six seven eight nine ten eleven twelve…" in response
    assert "thirteen" not in response
