# Speech emotion demo

This app uses **text and audio** from English speech. It predicts one MELD emotion and gives a short reply that quotes the input. It can transcribe a recording or use a transcript that you provide. Inference runs locally after a one-time model download.

## Run

Use Python 3.12. Clone this repository, open its folder, and run these commands. The trained emotion classifiers are included. Download the pretrained encoders once.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
export HF_HOME="$PWD/.cache/huggingface"
python scripts/download_models.py
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
meld-emotion serve
```

Open [the app](http://127.0.0.1:8000). Allow microphone access. Click **Record microphone**, speak, then click **Stop & analyze**. Record for 0.1 to 30 seconds. You can also upload a WAV file of up to 8 MiB. Uploads support one or two channels and sample rates from 8 to 48 kHz.

Use **Edit transcript** to fix any words. Predictions can be wrong. The model relies more on words than tone.

## How the model works

1. **Speech to text:** wav2vec 2.0 transcribes your recording. You can also supply the words yourself.
2. **Text features:** MiniLM turns the words into 384 numbers.
3. **Audio features:** wav2vec 2.0 turns the voice into 1,536 numbers. These describe the average and variation of its audio features.
4. **Emotion:** small logistic regression classifiers use the text, audio, or their joined features. The final result gives equal weight to the text-only and combined probabilities. This weight was selected on the development split.
5. **Reply:** a fixed sentence template uses the predicted emotion and quotes your input. Low confidence leads to a follow-up question.

The seven labels are neutral, surprise, fear, sadness, joy, disgust, and anger. The API returns the label, confidence, all seven probabilities, the reply, and the input status in JSON. Silent audio uses a text-only fallback when a transcript is supplied. Automatic transcription can fail when no speech is recognized.

The two pretrained encoders stay fixed. Only the small classifiers, feature scaling, and calibration are fitted on MELD. This keeps training small and makes each stage easy to inspect. Combining text and audio lets tone affect the result. The extra text prediction limits the effect of a weak audio classifier. A fixed reply template is fast and predictable, but its replies are less flexible than generated conversation.

The complete local model uses the following conservative parameter count. It includes all three classifiers, their scaling and calibration values, and the speech output head. The speech output head shares the audio encoder.

| Component | Parameters or stored model values |
|---|---:|
| MiniLM text encoder | 22,713,216 |
| wav2vec 2.0 audio encoder | 94,370,944 |
| Speech recognition output head | 24,608 |
| Emotion classifiers, scaling, and calibration | 34,585 |
| Reply templates | 0 |
| **Total** | **117,143,353** |

This is below the 6 billion parameter limit. It does not use a remote model API. Vision, speech synthesis, and reinforcement learning are not implemented. They are optional extensions to this text-and-audio prototype.

## What real time means here

The app sends audio chunks while you record. It transcribes and predicts after you stop. Each utterance is independent. It does not produce partial emotion labels or remember earlier turns.

The local target is a **warm 95th-percentile reply time of at most 500 ms after the end of an utterance**. This is a project target, not a guarantee on every computer. It assumes one user, models already loaded, and English audio of 0.1 to 30 seconds. Cold model loading is measured separately. The server runs one inference request at a time. Another request receives a busy error and can be retried. Audio, message sizes, and idle connections have fixed limits.

## Test results

These scores use the same 2,608 MELD test clips with usable audio. They use the dataset's reference transcripts. They do not measure automatic transcription accuracy. Macro F1 gives each emotion equal weight.

| Input | Accuracy | Macro F1 |
|---|---:|---:|
| Text only | 47.81% | 37.55% |
| Audio only | 45.51% | 22.23% |
| Text and audio | 48.77% | 37.85% |

The gain over text alone is small. On the full 2,610-clip test replay, two clips use text fallback because their audio is too long. The macro F1 gain is 0.31 percentage points. A paired bootstrap over dialogues gives a 95% interval from -0.77 to +1.32 points. That interval includes zero, so this test does not establish a reliable improvement. See the [saved evaluation](reports/test_metrics.json).

- **Software checks:** 161 Python tests and 9 JavaScript interface tests passed locally on October 13, 2026. Lint and formatting checks passed. The JavaScript tests use browser adapters in Node. See [CI runs](https://github.com/paulyan678/meld-realtime-emotion/actions/workflows/ci.yml).
- **Real browser checks:** Chrome passed WAV upload, transcript correction, physical microphone capture, invalid-file handling, and retry. The checks found and fixed a stale-result display after new input or failure. Microphone testing verified capture and processing, not word accuracy. See the [browser check](reports/browser_smoke.json).
- **Automatic speech over HTTP and WebSocket:** 93 of 100 clips completed. Seven returned no recognized speech. Median reply time was **74 ms** and p95 was **201 ms**. The first HTTP request with model loading took **2,090 ms**. Warm times cover successful requests only, from the client's end event to the complete response on localhost. This met the 500 ms p95 target on this sample. See the [current speech check](reports/speech_interaction_current.json).
- **Complete offline speech resource check:** On an **Apple M4 Max, 48 GiB RAM, 16 logical CPU cores, and four Torch threads**, CPU inference completed 93 of 100 clips. Seven returned no recognized speech. Median time was **73 ms** and p95 was **204 ms**. Peak process memory sampled every 10 ms was **1.13 GiB**. No GPU was used. The first prediction with fresh model loading took **2,109 ms**. This timer includes WAV decoding, transcription, emotion, and reply generation. It excludes recording and transport. Checkpoint files were already read for hashing. The run used macOS 27.0.1 and Python 3.12.13. See the [full resource report](reports/runtime_speech_cpu.json).

The older [CPU benchmark](reports/runtime_cpu.json) and [MPS benchmark](reports/runtime_mps.json) use supplied transcripts. Their memory and parameter totals do not include the later speech recognition head. They are historical measurements, not current full speech resource results.

These tests use TV dialogue from MELD. New speakers, background noise, accents, and different recording devices can change the results. A recognized transcript is not proof that the words are correct. A high emotion score is not proof of a person's actual feelings. The prototype does not yet measure automatic transcription word error rate or downstream emotion accuracy with automatic transcripts.

## Real example

This [recorded run](reports/interaction.json) used the 2.79-second MELD clip `dev/dia0_utt0`. It supplied both the audio and this transcript:

> Oh my God, he’s lost it. He’s totally lost it.

The model returned **sadness** with **49.6% confidence**. The dataset label was also sadness. The reply was:

> About “Oh my God, he’s lost it. He’s totally lost it.”: That sounds difficult. Would you like to talk about it?

The text classifier predicted sadness. The audio classifier predicted joy. The final result used both. This example uses a supplied transcript; automatic transcription can produce different words and results.

To reproduce it, install FFmpeg and curl first. The data script downloads the full MELD archive even when only the development split is requested. The archive is about **10.9 GB**. Allow about **35 GB of free disk space** for full preparation, including archives and extracted files. This is separate from the encoder cache.

```bash
python scripts/prepare_data.py --data-dir data/meld --splits dev --allow-missing
meld-emotion predict \
  --audio data/meld/audio/dev/dia0_utt0.wav \
  --text "Oh my God, he’s lost it. He’s totally lost it."
```

The command prints JSON. Its main fields should match this saved result, apart from small numeric differences across devices:

```json
{
  "emotion": "sadness",
  "confidence": 0.49643702718922755,
  "modality_status": "text_and_audio",
  "text_emotion": "sadness",
  "audio_emotion": "joy",
  "response": "About “Oh my God, he’s lost it. He’s totally lost it.”: That sounds difficult. Would you like to talk about it?"
}
```

The full output also includes seven emotion probabilities and runtime details. To try speech recognition, upload the same WAV in the app and leave **Automatic** selected. The saved automatic run recognized different words: `I GHOT HE'S LOST THAT HE SOLLY LOST HIM`. It still predicted sadness, with 57.6% confidence. This shows why the transcript correction step matters. See the [automatic example](reports/speech_interaction.json).

<details>
<summary>Reproduce training, evaluation, and software checks</summary>

Run these commands from the repository folder in the same activated environment. Keep the `HF_HOME` and offline exports from the setup commands. Data preparation still needs a network connection. It downloads pinned data files and records their hashes. `--allow-missing` keeps explicitly failed audio rows in the audit instead of treating them as valid audio.

```bash
python -m pip install -e '.[dev]'
python scripts/prepare_data.py --data-dir data/meld --allow-missing
for split in train dev test; do
  python -m meld_emotion.features \
    --manifest "data/meld/manifests/$split.jsonl" \
    --output "work/repro/$split.npz" \
    --cache .cache/meld-features --device cpu
done
python -m meld_emotion.training \
  --train work/repro/train.npz --dev work/repro/dev.npz \
  --output work/repro/model --report work/repro/development.json
python -m meld_emotion.evaluate \
  --features work/repro/test.npz --model-dir work/repro/model \
  --output work/repro/test_metrics.json
ruff check .
ruff format --check .
pytest -q
node --test tests/test_ui.cjs
```

The interface tests require Node.js 20 or later. Training uses the official training split. Model selection and calibration use the development split. Final scores use the test split. Generated files go into `work/repro`; the supplied model is kept intact. Inspect `data/meld/data_summary.json` for excluded or missing audio.

To repeat the real HTTP and WebSocket check, start `meld-emotion serve`. In a second terminal with the same environment, run:

```bash
mkdir -p work/repro
python scripts/check_interaction.py \
  --manifest data/meld/manifests/dev.jsonl --samples 100 \
  --automatic --output work/repro/speech_interaction.json
```

To measure the full local model without a server, run:

```bash
python scripts/benchmark.py \
  --manifest data/meld/manifests/dev.jsonl --samples 100 \
  --device cpu --automatic --output work/repro/runtime_speech_cpu.json
```

This benchmark blocks Python socket connections and uses only cached models. It records code and model hashes, hardware, all failed attempts, and sampled peak process memory. Its warm timer covers WAV decoding, transcription, emotion, and reply generation. It excludes recording and transport. Its cold timer includes fresh model initialization. Checkpoint files have already been read for hashing, so this is not a cold-disk measurement.

The HTTP and WebSocket check sends successive audio chunks and checks the final emotion and reply. It measures the local server path. It does not exercise microphone permission or a physical recording device.

</details>

## Credits and development record

Uses the [MELD dataset](https://github.com/declare-lab/MELD) by Poria et al. (2019),
[MiniLM](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), and
[wav2vec 2.0](https://huggingface.co/facebook/wav2vec2-base-960h).
Upstream licenses apply to each external component. Dataset media and pretrained
encoder weights are downloaded separately.

The implementation also uses PyTorch, Transformers, scikit-learn, NumPy, SciPy, SoundFile, FastAPI, and other dependencies listed in `pyproject.toml`. Codex assisted with implementation, review, testing, and documentation.

About 20 hours of development were spread across two weeks, as reported by the author. Commit dates were reconstructed and do not establish the timebox. The elapsed calendar period was longer than two to three days.
