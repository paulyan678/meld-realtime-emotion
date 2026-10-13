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

## Credits

Uses the [MELD dataset](https://github.com/declare-lab/MELD) by Poria et al. (2019),
[MiniLM](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), and
[wav2vec 2.0](https://huggingface.co/facebook/wav2vec2-base-960h).
Upstream licenses apply to each external component. Dataset media and pretrained
encoder weights are downloaded separately.
