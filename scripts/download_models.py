"""Download pinned encoders and speech tokenizer; runtime is offline by default."""

from meld_emotion.encoders import Encoders

if __name__ == "__main__":
    models = Encoders(local_files_only=False)
    models.enable_transcription()
    print(models.parameter_counts)
    print("Pinned encoders and transcription assets cached. Runtime can use HF_HUB_OFFLINE=1.")
