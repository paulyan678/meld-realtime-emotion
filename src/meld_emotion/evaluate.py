"""One final held-out evaluation after model selection is frozen."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

from .constants import AUDIO_REVISION, LABELS, TEXT_REVISION
from .inference import Heads
from .training import metrics, modality_features


def evaluate(features, model_dir, output):
    data = np.load(features, allow_pickle=False)
    if not all(str(i).startswith("test/") for i in data["ids"]):
        raise ValueError("Final evaluation requires official test/ IDs")
    if (
        str(data["text_revision"]) != TEXT_REVISION
        or str(data["audio_revision"]) != AUDIO_REVISION
        or str(data["feature_version"]) != "meanstd6s-v1"
    ):
        raise ValueError("Feature revision mismatch")
    heads = Heads(model_dir)
    probabilities = {
        name: heads.probabilities(modality_features(data, name), name)
        for name in ("text", "audio", "fusion")
    }
    probabilities["fusion_head"] = probabilities["fusion"]
    probabilities["fusion"] = heads.multimodal_probabilities(data["text"], data["audio"])
    # Mirror the deployed silence fallback; record raw fusion separately.
    probabilities["replay_with_fallback"] = probabilities["fusion"].copy()
    eligible = data["audio_available"] & (data["durations"] <= 30)
    silent = (data["rms"] < 1e-5) | ~eligible
    probabilities["replay_with_fallback"][silent] = probabilities["text"][silent]
    y = data["labels"]
    # Never score a missing/overlong audio placeholder as a real modality input.
    results = {
        name: metrics(
            y if name in {"text", "replay_with_fallback"} else y[eligible],
            p if name in {"text", "replay_with_fallback"} else p[eligible],
        )
        for name, p in probabilities.items()
    }
    common = eligible
    common_results = {name: metrics(y[common], p[common]) for name, p in probabilities.items()}
    # Paired dialogue bootstrap respects within-dialogue dependence.
    dialogue_ids = np.array([str(i).rsplit("_utt", 1)[0] for i in data["ids"]])
    groups = [np.flatnonzero(dialogue_ids == i) for i in np.unique(dialogue_ids)]
    rng = np.random.default_rng(42)
    differences = []
    text_pred = probabilities["text"].argmax(1)
    multimodal_pred = probabilities["replay_with_fallback"].argmax(1)
    for _ in range(1000):
        indices = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        differences.append(
            f1_score(
                y[indices],
                multimodal_pred[indices],
                labels=range(7),
                average="macro",
                zero_division=0,
            )
            - f1_score(
                y[indices], text_pred[indices], labels=range(7), average="macro", zero_division=0
            )
        )
    report = {
        "split": "test",
        "metrics": results,
        "labels": LABELS,
        "common_available_audio_metrics": common_results,
        "missing_audio_count": int((~data["audio_available"]).sum()),
        "overlong_audio_count": int((data["durations"] > 30).sum()),
        "evaluation_audio_policy": "Dataset replay treats absent or >30s audio as missing and uses text fallback; live raw >30s uploads are rejected.",
        "total_text_fallback_count": int(silent.sum()),
        "silent_audio_count": int(((data["rms"] < 1e-5) & eligible).sum()),
        "heads_sha256": hashlib.sha256((Path(model_dir) / "heads.npz").read_bytes()).hexdigest(),
        "paired_dialogue_bootstrap": {
            "replicates": 1000,
            "seed": 42,
            "macro_f1_difference_replay_minus_text": results["replay_with_fallback"]["macro_f1"]
            - results["text"]["macro_f1"],
            "percentile_95_ci": np.percentile(differences, [2.5, 97.5]).tolist(),
        },
    }
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    with path.with_name("test_predictions.csv").open("w") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(
            [
                "id",
                "true_emotion",
                "text_emotion",
                "audio_emotion",
                "fusion_emotion",
                "replay_emotion",
                "confidence",
            ]
        )
        for i, identifier in enumerate(data["ids"]):
            writer.writerow(
                [
                    identifier,
                    LABELS[y[i]],
                    *[
                        LABELS[probabilities[name][i].argmax()]
                        if name in {"text", "replay_with_fallback"} or eligible[i]
                        else "excluded"
                        for name in ["text", "audio", "fusion", "replay_with_fallback"]
                    ],
                    float(probabilities["replay_with_fallback"][i].max()),
                ]
            )
    print(
        json.dumps(
            {
                k: {m: v[m] for m in ["accuracy", "macro_f1", "weighted_f1"]}
                for k, v in results.items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", required=True)
    parser.add_argument("--model-dir", default="artifacts/model")
    parser.add_argument("--output", default="reports/test_metrics.json")
    args = parser.parse_args()
    evaluate(args.features, args.model_dir, args.output)
