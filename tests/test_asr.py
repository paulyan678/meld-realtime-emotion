"""ASR tests use local synthetic logits and a tiny tokenizer, without downloads."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from safetensors.torch import save_file
from transformers import Wav2Vec2CTCTokenizer

from meld_emotion.constants import AUDIO_MODEL, AUDIO_REVISION
from meld_emotion.encoders import Encoders


@pytest.fixture
def tokenizer(tmp_path):
    vocabulary = {"<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3, "|": 4, "A": 5, "B": 6}
    path = tmp_path / "vocab.json"
    path.write_text(json.dumps(vocabulary))
    return Wav2Vec2CTCTokenizer(str(path))


class SyntheticBackbone:
    config = SimpleNamespace(
        conv_kernel=[10, 3, 3, 3, 3, 2, 2],
        conv_stride=[5, 2, 2, 2, 2, 2, 2],
        hidden_size=7,
        vocab_size=7,
    )

    def __init__(self, tokens):
        self.tokens = tokens
        self.input_lengths = []

    def __call__(self, samples):
        call = len(self.input_lengths)
        start = max(0, call * 64000 - 16000)
        length = samples.shape[-1]
        self.input_lengths.append(length)
        centers = start + 200 + np.arange((length - 400) // 320 + 1) * 320
        ids = self.tokens(centers)
        logits = torch.full((1, len(centers), 7), -10.0)
        for index, token in enumerate(ids):
            logits[0, index, int(token)] = 10.0
        return SimpleNamespace(last_hidden_state=logits)


def make_encoder(tokenizer, token_function):
    encoder = object.__new__(Encoders)
    encoder.device = "cpu"
    encoder.local_files_only = True
    encoder.audio_model = SyntheticBackbone(token_function)
    encoder.asr_head = torch.nn.Identity()
    encoder.asr_tokenizer = tokenizer
    encoder.parameter_counts = {"asr_ctc_head": 56}
    return encoder


def waveform(seconds, rate=16000):
    time = np.arange(int(seconds * rate)) / rate
    return (0.1 * np.sin(2 * np.pi * 220 * time)).astype(np.float32)


def test_ctc_preserves_repeated_letters_separated_by_blank(tokenizer):
    sequence = [5, 5, 0, 5, 4, 6, 6]

    def tokens(centers):
        return [sequence[index] if index < len(sequence) else 0 for index in range(len(centers))]

    encoder = make_encoder(tokenizer, tokens)
    assert encoder.transcribe(waveform(1), 16000) == "AA B"


def test_ctc_collapse_spans_context_windows_without_duplicate_words(tokenizer):
    encoder = make_encoder(tokenizer, lambda centers: np.full(len(centers), 5))
    assert encoder.transcribe(waveform(30), 16000) == "A"
    assert len(encoder.audio_model.input_lengths) == 8
    assert max(encoder.audio_model.input_lengths) == 6 * 16000


def test_ctc_keeps_blank_at_window_boundary(tokenizer):
    def tokens(centers):
        return np.where((centers >= 63500) & (centers < 64000), 0, 5)

    encoder = make_encoder(tokenizer, tokens)
    assert encoder.transcribe(waveform(8), 16000) == "AA"


@pytest.mark.parametrize("token", [0, 1, 2, 3, 4])
def test_blank_special_or_whitespace_only_output_is_not_speech(tokenizer, token):
    encoder = make_encoder(tokenizer, lambda centers: np.full(len(centers), token))
    with pytest.raises(ValueError, match="No speech recognized"):
        encoder.transcribe(waveform(1), 16000)


@pytest.mark.parametrize("audio", [np.zeros(16000), np.full(16000, 0.3), waveform(31)])
def test_silence_and_overlong_input_rejected_before_model_load(tokenizer, audio):
    encoder = make_encoder(tokenizer, lambda centers: np.full(len(centers), 5))

    def unexpected_load():
        raise AssertionError("Invalid audio must be rejected before ASR initialization")

    encoder.enable_transcription = unexpected_load
    with pytest.raises(ValueError, match="No speech detected|at most 30"):
        encoder.transcribe(audio, 16000)


def test_transcription_resamples_before_audio_model(tokenizer):
    encoder = make_encoder(tokenizer, lambda centers: np.full(len(centers), 5))
    assert encoder.transcribe(waveform(1, rate=48000), 48000) == "A"
    assert encoder.audio_model.input_lengths == [16000]


def test_enable_loads_only_pinned_head_and_is_idempotent(tokenizer, tmp_path, monkeypatch):
    encoder = make_encoder(tokenizer, lambda centers: np.full(len(centers), 5))
    encoder.asr_head = None
    encoder.asr_tokenizer = None
    backbone = encoder.audio_model
    weights = torch.arange(49, dtype=torch.float32).reshape(7, 7)
    bias = torch.arange(7, dtype=torch.float32)
    checkpoint = tmp_path / "model.safetensors"
    save_file({"lm_head.weight": weights, "lm_head.bias": bias}, checkpoint)
    calls = []

    def download(repository, **kwargs):
        calls.append((repository, kwargs))
        return str(checkpoint)

    def load_tokenizer(repository, **kwargs):
        assert repository == AUDIO_MODEL
        assert kwargs == {"revision": AUDIO_REVISION, "local_files_only": True}
        return tokenizer

    monkeypatch.setattr("huggingface_hub.hf_hub_download", download)
    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", load_tokenizer)
    encoder.enable_transcription()
    encoder.enable_transcription()
    assert calls == [
        (
            AUDIO_MODEL,
            {
                "filename": "model.safetensors",
                "revision": AUDIO_REVISION,
                "local_files_only": True,
            },
        )
    ]
    assert encoder.audio_model is backbone
    assert encoder.asr_tokenizer is tokenizer
    assert torch.equal(encoder.asr_head.weight, weights)
    assert torch.equal(encoder.asr_head.bias, bias)
    assert sum(parameter.numel() for parameter in encoder.asr_head.parameters()) == 56
    assert all(not parameter.requires_grad for parameter in encoder.asr_head.parameters())
