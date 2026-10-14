# Speech emotion demo

I built this app to predict emotion from **text and audio** in English speech. My app predicts one MELD emotion and gives a short reply that quotes the input. I can let it transcribe a recording or provide a transcript myself. I run inference locally after a one-time model download.

## How I run it

I use Python 3.12. I clone this repository, open its folder, and run these commands. I include the trained emotion classifiers in the repository and download the pretrained encoders once.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
export HF_HOME="$PWD/.cache/huggingface"
python scripts/download_models.py
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
meld-emotion serve
```

I open [the app](http://127.0.0.1:8000) and allow microphone access. I click **Record microphone**, speak for 0.1 to 30 seconds, then click **Stop & analyze**. I can also upload a WAV file of up to 8 MiB, with one or two channels and a sample rate from 8 to 48 kHz.

I use **Edit transcript** to fix any words. My model's predictions can be wrong, and it relies more on words than tone.

## How I built the model

1. **Speech to text:** I use wav2vec 2.0 to transcribe my recording, or I supply the words myself.
2. **Text features:** I use MiniLM to turn the words into 384 numbers.
3. **Audio features:** I use wav2vec 2.0 to turn the voice into 1,536 numbers describing the average and variation of its audio features.
4. **Emotion:** I use small logistic regression classifiers with the text, audio, or their joined features. I give equal weight to the text-only and combined probabilities in the final result. I selected this weight on the development split.
5. **Reply:** I use a fixed sentence template that incorporates the predicted emotion and quotes the input. When confidence is low, I have the app ask a follow-up question.

I predict one of seven labels: neutral, surprise, fear, sadness, joy, disgust, and anger. My API returns the label, confidence, all seven probabilities, the reply, and the input status in JSON. I use a text-only fallback for silent audio when a transcript is supplied. Automatic transcription in my app can fail when no speech is recognized.

I keep the two pretrained encoders fixed and fit only the small classifiers, feature scaling, and calibration on MELD. This keeps my training small and makes each stage easy to inspect. I combine text and audio so tone can affect the result, and I include an extra text prediction to limit the effect of a weak audio classifier. I use a fixed reply template for speed and predictability, though its replies are less flexible than generated conversation.

I use the following conservative parameter count for my complete local model. I include all three classifiers, their scaling and calibration values, and the speech output head, which shares the audio encoder.

| Component | Parameters or stored model values |
|---|---:|
| MiniLM text encoder | 22,713,216 |
| wav2vec 2.0 audio encoder | 94,370,944 |
| Speech recognition output head | 24,608 |
| Emotion classifiers, scaling, and calibration | 34,585 |
| Reply templates | 0 |
| **Total** | **117,143,353** |

My model stays below the 6 billion parameter limit, and I do not use a remote model API. I have not implemented vision, speech synthesis, or reinforcement learning. I consider these optional extensions to my text-and-audio prototype.

## What I mean by real time

I send audio chunks from the app while recording, then transcribe and predict after recording stops. I process each utterance independently; my app does not produce partial emotion labels or remember earlier turns.

I target a **warm 95th-percentile reply time of at most 500 ms after the end of an utterance** locally. I treat this as a project target, not a guarantee on every computer. I assume one user, models already loaded, and English audio of 0.1 to 30 seconds, and I measure cold model loading separately. I run one inference request at a time on the server. I return a busy error for another request so it can be retried, and I set fixed limits on audio, message sizes, and idle connections.

## My test results

I evaluated all three inputs on the same 2,608 MELD test clips with usable audio, using the dataset's reference transcripts. I did not measure automatic transcription accuracy in these scores. I report macro F1 to give each emotion equal weight.

| Input | Accuracy | Macro F1 |
|---|---:|---:|
| Text only | 47.81% | 37.55% |
| Audio only | 45.51% | 22.23% |
| Text and audio | 48.77% | 37.85% |

I found only a small gain over text alone. In my full 2,610-clip test replay, I used text fallback for two clips because their audio was too long. I measured a macro F1 gain of 0.31 percentage points. My paired bootstrap over dialogues gave a 95% interval from -0.77 to +1.32 points. Because that interval includes zero, I cannot conclude that this test establishes a reliable improvement. I saved the results in the [evaluation report](reports/test_metrics.json).

- **Software checks:** I ran 161 Python tests and 9 JavaScript interface tests locally on October 13, 2026, and all passed. My lint and formatting checks also passed. I use browser adapters in Node for the JavaScript tests. I also track [CI runs](https://github.com/paulyan678/meld-realtime-emotion/actions/workflows/ci.yml).
- **Real browser checks:** I verified WAV upload, transcript correction, physical microphone capture, invalid-file handling, and retry in Chrome. During these checks, I found and fixed a stale-result display after new input or failure. My microphone testing verified capture and processing, not word accuracy. I recorded the results in the [browser check](reports/browser_smoke.json).
- **Automatic speech over HTTP and WebSocket:** I completed 93 of 100 clips; seven returned no recognized speech. I measured a median reply time of **74 ms** and a p95 of **201 ms**. My first HTTP request with model loading took **2,090 ms**. I measured warm times for successful requests only, from the client's end event to the complete response on localhost. I met my 500 ms p95 target on this sample and saved the results in the [current speech check](reports/speech_interaction_current.json).
- **Complete offline speech resource check:** I ran CPU inference on an **Apple M4 Max, 48 GiB RAM, 16 logical CPU cores, and four Torch threads**, completing 93 of 100 clips; seven returned no recognized speech. I measured a median time of **73 ms** and a p95 of **204 ms**. I sampled peak process memory every 10 ms and recorded **1.13 GiB**. I did not use a GPU. My first prediction with fresh model loading took **2,109 ms**. I timed WAV decoding, transcription, emotion, and reply generation, excluding recording and transport. I had already read the checkpoint files for hashing. I used macOS 27.0.1 and Python 3.12.13, and saved the results in the [full resource report](reports/runtime_speech_cpu.json).

I used supplied transcripts in my older [CPU benchmark](reports/runtime_cpu.json) and [MPS benchmark](reports/runtime_mps.json). I did not include the later speech recognition head in their memory and parameter totals, so I keep these as historical measurements rather than current full speech resource results.

I tested on TV dialogue from MELD, so my results can change with new speakers, background noise, accents, and different recording devices. I do not treat a recognized transcript as proof that the words are correct, or a high emotion score as proof of a person's actual feelings. I have not yet measured automatic transcription word error rate or downstream emotion accuracy with automatic transcripts.

## An example from my app

In this [recorded run](reports/interaction.json), I used the 2.79-second MELD clip `dev/dia0_utt0` and supplied both the audio and this transcript:

> Oh my God, he’s lost it. He’s totally lost it.

My model returned **sadness** with **49.6% confidence**, matching the dataset's sadness label. My app gave this reply:

> About “Oh my God, he’s lost it. He’s totally lost it.”: That sounds difficult. Would you like to talk about it?

My text classifier predicted sadness, while my audio classifier predicted joy. I used both in the final result. I supplied a transcript for this example; automatic transcription can produce different words and results.

To reproduce this example, I install FFmpeg and curl first. My data script downloads the full MELD archive even when I request only the development split. The archive is about **10.9 GB**, so I allow about **35 GB of free disk space** for full preparation, including archives and extracted files, separately from the encoder cache.

```bash
python scripts/prepare_data.py --data-dir data/meld --splits dev --allow-missing
meld-emotion predict \
  --audio data/meld/audio/dev/dia0_utt0.wav \
  --text "Oh my God, he’s lost it. He’s totally lost it."
```

I get JSON output from this command. I expect its main fields to match my saved result below, apart from small numeric differences across devices:

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

I also include seven emotion probabilities and runtime details in the full output. To try speech recognition, I upload the same WAV in the app and leave **Automatic** selected. In my saved automatic run, the app recognized different words: `I GHOT HE'S LOST THAT HE SOLLY LOST HIM`. My model still predicted sadness, with 57.6% confidence. This is why I include a transcript correction step. I saved that run in the [automatic example](reports/speech_interaction.json).

<details>
<summary>How I reproduce training, evaluation, and software checks</summary>

I run these commands from the repository folder in the same activated environment, keeping the `HF_HOME` and offline exports from my setup commands. I still need a network connection for data preparation, which downloads pinned data files and records their hashes. I use `--allow-missing` to keep explicitly failed audio rows in the audit instead of treating them as valid audio.

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

My interface tests require Node.js 20 or later. I train on the official training split, select and calibrate my model on the development split, and report final scores on the test split. I write generated files to `work/repro` and keep the supplied model intact. I inspect `data/meld/data_summary.json` for excluded or missing audio.

To repeat my real HTTP and WebSocket check, I start `meld-emotion serve`. Then I run these commands in a second terminal with the same environment:

```bash
mkdir -p work/repro
python scripts/check_interaction.py \
  --manifest data/meld/manifests/dev.jsonl --samples 100 \
  --automatic --output work/repro/speech_interaction.json
```

To measure my full local model without a server, I run:

```bash
python scripts/benchmark.py \
  --manifest data/meld/manifests/dev.jsonl --samples 100 \
  --device cpu --automatic --output work/repro/runtime_speech_cpu.json
```

In this benchmark, I block Python socket connections and use only cached models. I record code and model hashes, hardware, all failed attempts, and sampled peak process memory. My warm timer covers WAV decoding, transcription, emotion, and reply generation, excluding recording and transport. My cold timer includes fresh model initialization. I have already read the checkpoint files for hashing, so I do not treat this as a cold-disk measurement.

In my HTTP and WebSocket check, I send successive audio chunks and check the final emotion and reply. I measure the local server path without exercising microphone permission or a physical recording device.

</details>

## Credits

I use the [MELD dataset](https://github.com/declare-lab/MELD) by Poria et al. (2019),
[MiniLM](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), and
[wav2vec 2.0](https://huggingface.co/facebook/wav2vec2-base-960h).
My use of each external component is subject to its upstream license. I download
dataset media and pretrained encoder weights separately.

I also use PyTorch, Transformers, scikit-learn, NumPy, SciPy, SoundFile, FastAPI, and the other dependencies listed in `pyproject.toml`. I used Codex to assist with implementation, review, testing, and documentation.
