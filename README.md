# Speech emotion demo

This is my local speech emotion demo. It uses **text and audio** from English speech to predict one MELD emotion and give a short reply that quotes the input. You can let the app transcribe a recording or provide a transcript yourself. After a one-time model download, inference runs locally.

## Getting started

Start with Python 3.12, clone this repository, and open its folder. I've included the trained emotion classifiers; the setup below downloads the pretrained encoders once:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
export HF_HOME="$PWD/.cache/huggingface"
python scripts/download_models.py
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
meld-emotion serve
```

Open [the app](http://127.0.0.1:8000) and allow microphone access. Click **Record microphone**, speak for 0.1 to 30 seconds, then click **Stop & analyze**. You can also upload a WAV file of up to 8 MiB, with one or two channels and a sample rate from 8 to 48 kHz.

Use **Edit transcript** to correct any words before interpreting the result. Keep in mind that predictions can be wrong, and the model relies more on words than tone.

## How the model works

1. **Speech to text:** wav2vec 2.0 transcribes your recording, or you can supply a transcript yourself.
2. **Text features:** MiniLM turns the words into 384 numbers.
3. **Audio features:** wav2vec 2.0 turns the voice into 1,536 numbers describing the average and variation of its audio features.
4. **Emotion:** small logistic regression classifiers use the text, audio, or their joined features. The final result gives equal weight to the text-only and combined probabilities, a weight selected on the development split.
5. **Reply:** a fixed sentence template incorporates the predicted emotion and quotes your input. Low confidence leads to a follow-up question.

The seven labels are neutral, surprise, fear, sadness, joy, disgust, and anger. If you're using the API, you'll receive the label, confidence, all seven probabilities, the reply, and the input status in JSON. Silent audio falls back to text when a transcript is supplied. Automatic transcription can fail when no speech is recognized.

I keep the two pretrained encoders fixed and fit only the small classifiers, feature scaling, and calibration on MELD. That keeps training small and makes each stage easy to inspect. Combining text and audio lets tone affect the result, while the extra text prediction limits the effect of a weak audio classifier. For replies, I use a fixed template for speed and predictability, at the cost of less flexible conversation.

For a conservative count of the complete local model, I include all three classifiers, their scaling and calibration values, and the speech output head. The speech output head shares the audio encoder:

| Component | Parameters or stored model values |
|---|---:|
| MiniLM text encoder | 22,713,216 |
| wav2vec 2.0 audio encoder | 94,370,944 |
| Speech recognition output head | 24,608 |
| Emotion classifiers, scaling, and calibration | 34,585 |
| Reply templates | 0 |
| **Total** | **117,143,353** |

This stays below the 6 billion parameter limit and requires no remote model API. Vision, speech synthesis, and reinforcement learning are optional extensions that I haven't implemented in this text-and-audio prototype.

## What I mean by real time

As you record, the app sends audio chunks to the server. Transcription and prediction happen after you stop. Each utterance is independent; the app doesn't produce partial emotion labels or remember earlier turns.

My local target is a **warm 95th-percentile reply time of at most 500 ms after the end of an utterance**. Treat this as a project target, not a guarantee on every computer. It assumes one user, models already loaded, and English audio of 0.1 to 30 seconds. Cold model loading is measured separately. The server handles one inference request at a time, so retry if another request receives a busy error. Audio, message sizes, and idle connections have fixed limits.

## Test results and limitations

To compare the three inputs, I evaluated them on the same 2,608 MELD test clips with usable audio and the dataset's reference transcripts. These scores don't measure automatic transcription accuracy. Macro F1 gives each emotion equal weight.

| Input | Accuracy | Macro F1 |
|---|---:|---:|
| Text only | 47.81% | 37.55% |
| Audio only | 45.51% | 22.23% |
| Text and audio | 48.77% | 37.85% |

The gain over text alone is small. In the full 2,610-clip test replay, two clips use text fallback because their audio is too long. The macro F1 gain is 0.31 percentage points, with a 95% interval from -0.77 to +1.32 points in a paired bootstrap over dialogues. Since that interval includes zero, I wouldn't interpret this test as establishing a reliable improvement. You can inspect the [evaluation report](reports/test_metrics.json) for the saved results.

- **Software checks:** All 161 Python tests and 9 JavaScript interface tests passed locally on October 13, 2026, along with lint and formatting checks. The JavaScript tests use browser adapters in Node. See the [CI runs](https://github.com/paulyan678/meld-realtime-emotion/actions/workflows/ci.yml).
- **Real browser checks:** Chrome passed WAV upload, transcript correction, physical microphone capture, invalid-file handling, and retry. These checks also caught a stale-result display after new input or failure, which is now fixed. Microphone testing verified capture and processing, not word accuracy. See the [browser check](reports/browser_smoke.json).
- **Automatic speech over HTTP and WebSocket:** 93 of 100 clips completed; seven returned no recognized speech. Median reply time was **74 ms** and p95 was **201 ms**. The first HTTP request with model loading took **2,090 ms**. Warm times cover successful requests only, from the client's end event to the complete response on localhost. This sample met the 500 ms p95 target. See the [current speech check](reports/speech_interaction_current.json).
- **Complete offline speech resource check:** CPU inference completed 93 of 100 clips on an **Apple M4 Max, 48 GiB RAM, 16 logical CPU cores, and four Torch threads**; seven returned no recognized speech. Median time was **73 ms** and p95 was **204 ms**. Peak process memory, sampled every 10 ms, was **1.13 GiB**, with no GPU used. The first prediction with fresh model loading took **2,109 ms**. This timer includes WAV decoding, transcription, emotion, and reply generation, but excludes recording and transport. Checkpoint files had already been read for hashing. The run used macOS 27.0.1 and Python 3.12.13. See the [full resource report](reports/runtime_speech_cpu.json).

When comparing benchmarks, note that my older [CPU benchmark](reports/runtime_cpu.json) and [MPS benchmark](reports/runtime_mps.json) use supplied transcripts. Their memory and parameter totals exclude the later speech recognition head, so treat them as historical measurements rather than current full speech resource results.

These tests use TV dialogue from MELD. Your results may differ with new speakers, background noise, accents, and different recording devices. A recognized transcript doesn't prove that the words are correct, and a high emotion score doesn't prove a person's actual feelings. I haven't yet measured automatic transcription word error rate or downstream emotion accuracy with automatic transcripts.

## Try an example

Here's a [recorded run](reports/interaction.json) you can reproduce with the 2.79-second MELD clip `dev/dia0_utt0`. It uses both the audio and this supplied transcript:

> Oh my God, he’s lost it. He’s totally lost it.

The model returned **sadness** with **49.6% confidence**, matching the dataset's sadness label, and gave this reply:

> About “Oh my God, he’s lost it. He’s totally lost it.”: That sounds difficult. Would you like to talk about it?

The text classifier predicted sadness, while the audio classifier predicted joy; the final result used both. Keep in mind that this example uses a supplied transcript. Automatic transcription can produce different words and results.

To try it yourself, install FFmpeg and curl first. The data script downloads the full MELD archive even when you request only the development split. The archive is about **10.9 GB**, so allow about **35 GB of free disk space** for full preparation, including archives and extracted files, in addition to the encoder cache.

```bash
python scripts/prepare_data.py --data-dir data/meld --splits dev --allow-missing
meld-emotion predict \
  --audio data/meld/audio/dev/dia0_utt0.wav \
  --text "Oh my God, he’s lost it. He’s totally lost it."
```

The command prints JSON. Its main fields should match the saved result below, apart from small numeric differences across devices:

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

The full output also includes seven emotion probabilities and runtime details. To try speech recognition, upload the same WAV in the app and leave **Automatic** selected. In the [automatic example](reports/speech_interaction.json), the app recognized different words: `I GHOT HE'S LOST THAT HE SOLLY LOST HIM`. It still predicted sadness, with 57.6% confidence. This is why I include a transcript correction step.

<details>
<summary>Reproduce training, evaluation, and software checks</summary>

Run the following commands from the repository folder in the same activated environment, keeping the `HF_HOME` and offline exports from setup. You'll still need a network connection for data preparation, which downloads pinned data files and records their hashes. The `--allow-missing` flag keeps explicitly failed audio rows in the audit instead of treating them as valid audio.

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

The interface tests require Node.js 20 or later. Training uses the official training split, model selection and calibration use the development split, and final scores use the test split. Generated files go into `work/repro`, leaving the supplied model intact. Check `data/meld/data_summary.json` for excluded or missing audio.

To repeat the HTTP and WebSocket check, start `meld-emotion serve`, then run these commands in a second terminal with the same environment:

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

This benchmark blocks Python socket connections and uses only cached models. It records code and model hashes, hardware, all failed attempts, and sampled peak process memory. The warm timer covers WAV decoding, transcription, emotion, and reply generation, excluding recording and transport. The cold timer includes fresh model initialization. Since checkpoint files have already been read for hashing, this isn't a cold-disk measurement.

The HTTP and WebSocket check sends successive audio chunks and checks the final emotion and reply. Use it to measure the local server path; it doesn't exercise microphone permission or a physical recording device.

</details>

## Credits

This project uses the [MELD dataset](https://github.com/declare-lab/MELD) by Poria et al. (2019),
[MiniLM](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), and
[wav2vec 2.0](https://huggingface.co/facebook/wav2vec2-base-960h).
Upstream licenses apply to each external component. Dataset media and pretrained
encoder weights are downloaded separately.

The implementation uses PyTorch, Transformers, scikit-learn, NumPy, SciPy, SoundFile, FastAPI, and the other dependencies listed in `pyproject.toml`. I used Codex to assist with implementation, review, testing, and documentation.
