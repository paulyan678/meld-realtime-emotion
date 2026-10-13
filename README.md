# Speech emotion demo

This app transcribes English speech. It uses your words and voice to predict an emotion and give a short reply. Everything runs locally.

## Run

Use Python 3.12. Run these commands from this folder. Download the models once.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
export HF_HOME="$PWD/.cache/huggingface"
python scripts/download_models.py
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
meld-emotion serve
```

Open [the app](http://127.0.0.1:8000). Click **Record microphone**, speak, then click **Stop & analyze**. Record up to 30 seconds. You can also upload a WAV file.

Use **Edit transcript** to fix any words. Predictions can be wrong. The model relies more on words than tone.

## How the model works

1. **Speech to text:** wav2vec 2.0 transcribes your recording. You can also supply the words yourself.
2. **Text features:** MiniLM turns the words into 384 numbers.
3. **Audio features:** wav2vec 2.0 turns the voice into 1,536 numbers. These describe the average and variation of its audio features.
4. **Emotion:** small logistic regression classifiers use the text, audio, or both. The final result averages the text-only and combined predictions.
5. **Reply:** a fixed sentence template uses the predicted emotion and quotes your input. Low confidence leads to a follow-up question.

The two pretrained encoders stay fixed. Only the small classifiers are trained on MELD. The seven labels are neutral, surprise, fear, sadness, joy, disgust, and anger. Each recording is analyzed on its own, after you stop speaking.

## Test results

These scores use the same 2,608 MELD test clips with usable audio. They use the dataset's reference transcripts. They do not measure automatic transcription accuracy. Macro F1 gives each emotion equal weight.

| Input | Accuracy | Macro F1 |
|---|---:|---:|
| Text only | 47.81% | 37.55% |
| Audio only | 45.51% | 22.23% |
| Text and audio | 48.77% | 37.85% |

The gain over text alone is small. See the [saved evaluation](reports/test_metrics.json).

- **Software checks:** 154 Python tests and 7 browser-interface tests passed in [CI](https://github.com/paulyan678/meld-realtime-emotion-clean/actions/runs/38481695202).
- **Automatic speech check:** 93 of 100 clips completed. Seven returned no recognized speech. On the local CPU, the median reply time was 75 ms. The 95th percentile was 208 ms. These times start after recording ends and cover successful requests only. See the [saved speech check](reports/speech_interaction.json).

## Real example

This [recorded run](reports/interaction.json) used the 2.79-second MELD clip `dev/dia0_utt0`. It supplied both the audio and this transcript:

> Oh my God, he’s lost it. He’s totally lost it.

The model returned **sadness** with **49.6% confidence**. The dataset label was also sadness. The reply was:

> About “Oh my God, he’s lost it. He’s totally lost it.”: That sounds difficult. Would you like to talk about it?

The text classifier predicted sadness. The audio classifier predicted joy. The final result used both. This example uses a supplied transcript; automatic transcription can produce different words and results.

## Credits

Uses the [MELD dataset](https://github.com/declare-lab/MELD) by Poria et al. (2019),
[MiniLM](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), and
[wav2vec 2.0](https://huggingface.co/facebook/wav2vec2-base-960h).
Upstream licenses apply to each external component. Dataset media and pretrained
encoder weights are downloaded separately.
