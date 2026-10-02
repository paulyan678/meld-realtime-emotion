"""Exactly the same raw-input feature extraction in training and online inference."""

import math
import os

import numpy as np
from scipy.signal import resample_poly

from .constants import (
    AUDIO_CHUNK_SECONDS,
    AUDIO_MODEL,
    AUDIO_REVISION,
    MAX_AUDIO_SECONDS,
    SAMPLE_RATE,
    TEXT_MAX_TOKENS,
    TEXT_MODEL,
    TEXT_REVISION,
)


def prepare_audio(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    x = np.asarray(audio, dtype=np.float32)
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all():
        raise ValueError("Audio must be nonempty, finite, mono samples.")
    if not isinstance(sample_rate, (int, np.integer)) or not 8000 <= sample_rate <= 96000:
        raise ValueError("Sample rate must be an integer between 8000 and 96000 Hz.")
    if np.max(np.abs(x)) > 1.01:
        raise ValueError("Audio must be normalized floating-point PCM in [-1, 1].")
    if sample_rate != SAMPLE_RATE:
        divisor = math.gcd(int(sample_rate), SAMPLE_RATE)
        x = resample_poly(x, SAMPLE_RATE // divisor, int(sample_rate) // divisor)
    return np.asarray(x, dtype=np.float32)


class Encoders:
    def __init__(self, device="cpu", local_files_only=True):
        import torch
        from transformers import AutoConfig, AutoModel, AutoTokenizer

        if device not in {"cpu", "mps", "cuda"}:
            raise ValueError("device must be cpu, mps, or cuda")
        torch.set_num_threads(int(os.environ.get("MELD_TORCH_THREADS", "4")))
        self.device = device
        self.local_files_only = local_files_only
        self.asr_head = None
        self.asr_tokenizer = None
        self.tokenizer = AutoTokenizer.from_pretrained(
            TEXT_MODEL, revision=TEXT_REVISION, local_files_only=local_files_only
        )
        self.text_model = (
            AutoModel.from_pretrained(
                TEXT_MODEL,
                revision=TEXT_REVISION,
                local_files_only=local_files_only,
                use_safetensors=True,
            )
            .eval()
            .to(device)
        )
        audio_config = AutoConfig.from_pretrained(
            AUDIO_MODEL,
            revision=AUDIO_REVISION,
            local_files_only=local_files_only,
        )
        # Disable training-only augmentation; its random mask token is never used
        # during evaluation and is absent from the released ASR checkpoint.
        audio_config.mask_time_prob = 0.0
        audio_config.mask_feature_prob = 0.0
        self.audio_model = (
            AutoModel.from_pretrained(
                AUDIO_MODEL,
                revision=AUDIO_REVISION,
                local_files_only=local_files_only,
                use_safetensors=True,
                config=audio_config,
            )
            .eval()
            .to(device)
        )
        for model in (self.text_model, self.audio_model):
            model.requires_grad_(False)
        self.parameter_counts = {
            "text_encoder": sum(p.numel() for p in self.text_model.parameters()),
            "audio_encoder": sum(p.numel() for p in self.audio_model.parameters()),
            # Count the optional, lazily loaded output head conservatively even
            # before the first transcription. The tokenizer has no learned weights.
            "asr_ctc_head": audio_config.vocab_size * (audio_config.hidden_size + 1),
        }

    def enable_transcription(self):
        """Load the pinned CTC output head, reusing the existing audio backbone."""
        if self.asr_head is not None:
            return
        import torch
        from huggingface_hub import hf_hub_download
        from safetensors import safe_open
        from transformers import AutoTokenizer

        # Read just two tensors. Loading Wav2Vec2ForCTC here would create another
        # copy of the 94M-parameter backbone and waste memory.
        checkpoint = hf_hub_download(
            AUDIO_MODEL,
            filename="model.safetensors",
            revision=AUDIO_REVISION,
            local_files_only=self.local_files_only,
        )
        with safe_open(checkpoint, framework="pt", device="cpu") as tensors:
            weight = tensors.get_tensor("lm_head.weight")
            bias = tensors.get_tensor("lm_head.bias")
        config = self.audio_model.config
        if tuple(weight.shape) != (config.vocab_size, config.hidden_size) or tuple(bias.shape) != (
            config.vocab_size,
        ):
            raise ValueError("ASR head dimensions do not match the pinned audio backbone.")
        head = torch.nn.Linear(config.hidden_size, config.vocab_size)
        head.load_state_dict({"weight": weight, "bias": bias})
        head.eval().requires_grad_(False).to(self.device)
        if sum(p.numel() for p in head.parameters()) != self.parameter_counts["asr_ctc_head"]:
            raise ValueError("ASR parameter count differs from the reserved budget.")
        tokenizer = AutoTokenizer.from_pretrained(
            AUDIO_MODEL,
            revision=AUDIO_REVISION,
            local_files_only=self.local_files_only,
        )
        # Publish the head last so a failed tokenizer load can be retried safely.
        self.asr_tokenizer = tokenizer
        self.asr_head = head

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        """Greedy English CTC decoding after an utterance, entirely on this device.

        Up to six-second windows include one second of context on each side.
        Keep only each window's central frames and collapse CTC tokens once over
        the complete utterance, preserving repeated letters separated by blanks.
        This is bounded-window inference, not a causal streaming recognizer.
        """
        import torch

        x = prepare_audio(audio, sample_rate)
        if len(x) > MAX_AUDIO_SECONDS * SAMPLE_RATE:
            raise ValueError(f"Audio must be at most {MAX_AUDIO_SECONDS} seconds.")
        if np.std(x) < 1e-5:
            raise ValueError("No speech detected. Please speak clearly and try again.")
        self.enable_transcription()
        config = self.audio_model.config
        # Locate output-frame centers from the checkpoint's convolution stack.
        step, receptive_field = 1, 1
        for kernel, stride in zip(config.conv_kernel, config.conv_stride, strict=True):
            receptive_field += (kernel - 1) * step
            step *= stride
        context = SAMPLE_RATE
        core_size = (AUDIO_CHUNK_SECONDS - 2) * SAMPLE_RATE
        token_ids = []
        for core_start in range(0, len(x), core_size):
            core_end = min(core_start + core_size, len(x))
            start = max(0, core_start - context)
            end = min(len(x), core_end + context)
            chunk = x[start:end]
            if len(chunk) < receptive_field:
                chunk = np.pad(chunk, (0, receptive_field - len(chunk)))
            chunk = (chunk - chunk.mean()) / np.sqrt(chunk.var() + 1e-7)
            tensor = torch.from_numpy(chunk.copy()).unsqueeze(0).to(self.device)
            with torch.inference_mode():
                hidden = self.audio_model(tensor).last_hidden_state[0]
                predicted = self.asr_head(hidden).argmax(dim=-1).cpu().numpy()
            centers = start + receptive_field / 2 + np.arange(len(predicted)) * step
            keep = (centers >= core_start) & (centers < core_end)
            token_ids.extend(predicted[keep].tolist())
        # Do not skip special tokens before CTC collapse: removing the blank
        # first would incorrectly turn A, blank, A into a single A.
        special_ids = set(self.asr_tokenizer.all_special_ids)
        token_ids = [
            self.asr_tokenizer.pad_token_id if token in special_ids else token
            for token in token_ids
        ]
        transcript = " ".join(self.asr_tokenizer.decode(token_ids).split())
        if not any(letter.isalpha() for letter in transcript):
            raise ValueError("No speech recognized. Please speak clearly and try again.")
        return transcript

    def text(self, texts: list[str]) -> np.ndarray:
        import torch

        tokens = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=TEXT_MAX_TOKENS,
            return_tensors="pt",
        ).to(self.device)
        with torch.inference_mode():
            hidden = self.text_model(**tokens).last_hidden_state
            mask = tokens["attention_mask"].unsqueeze(-1)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
            pooled = torch.nn.functional.normalize(pooled, dim=1)
        return pooled.cpu().numpy().astype(np.float32)

    def audio(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        import torch

        x = prepare_audio(audio, sample_rate)
        frames = []
        chunk_size = AUDIO_CHUNK_SECONDS * SAMPLE_RATE
        # Independent nonoverlapping windows bound encoder attention memory. We do
        # not call the bidirectional wav2vec backbone a causal streaming model.
        for start in range(0, len(x), chunk_size):
            chunk = x[start : start + chunk_size]
            if len(chunk) < 400:
                chunk = np.pad(chunk, (0, 400 - len(chunk)))
            chunk = (chunk - chunk.mean()) / np.sqrt(chunk.var() + 1e-7)
            tensor = torch.from_numpy(chunk.copy()).unsqueeze(0).to(self.device)
            with torch.inference_mode():
                hidden = self.audio_model(tensor).last_hidden_state[0]
            frames.append(hidden.cpu().numpy())
        features = np.concatenate(frames)
        return np.concatenate([features.mean(0), features.std(0)]).astype(np.float32)
