"""Fit heads on training data and select/calibrate using development data only."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
)
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .constants import AUDIO_REVISION, LABELS, TEXT_REVISION


def metrics(y, probabilities):
    prediction = probabilities.argmax(1)
    return {
        "n": len(y),
        "accuracy": accuracy_score(y, prediction),
        "macro_f1": f1_score(y, prediction, average="macro", labels=range(7), zero_division=0),
        "weighted_f1": f1_score(y, prediction, average="weighted", zero_division=0),
        "log_loss": log_loss(y, probabilities, labels=range(7)),
        "per_class": classification_report(
            y, prediction, labels=range(7), target_names=LABELS, output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(y, prediction, labels=range(7)).tolist(),
    }


def modality_features(data, modality):
    return (
        np.concatenate([data["text"], data["audio"]], axis=1)
        if modality == "fusion"
        else data[modality]
    )


def fit(train_path, dev_path, output, report):
    train = np.load(train_path, allow_pickle=False)
    dev = np.load(dev_path, allow_pickle=False)
    if not all(str(i).startswith("train/") for i in train["ids"]):
        raise ValueError("Training file must contain only train/ IDs")
    if not all(str(i).startswith("dev/") for i in dev["ids"]):
        raise ValueError("Selection file must contain only dev/ IDs")
    if set(train["ids"]) & set(dev["ids"]):
        raise ValueError("Split overlap")
    for data in (train, dev):
        if (
            str(data["text_revision"]) != TEXT_REVISION
            or str(data["audio_revision"]) != AUDIO_REVISION
            or str(data["feature_version"]) != "meanstd6s-v1"
        ):
            raise ValueError("Feature revisions do not match the current encoders")
        if not np.isfinite(data["text"]).all() or not np.isfinite(data["audio"]).all():
            raise ValueError("Non-finite features")
    weights, results, selected = {}, {}, {}
    y, y_dev = train["labels"], dev["labels"]
    if set(y) != set(range(7)):
        raise ValueError("All seven classes are required in training")
    with threadpool_limits(limits=4):
        for modality in ["text", "audio", "fusion"]:
            train_mask = (
                np.ones(len(y), dtype=bool)
                if modality == "text"
                else train["audio_available"] & (train["durations"] <= 30)
            )
            dev_mask = (
                np.ones(len(y_dev), dtype=bool)
                if modality == "text"
                else dev["audio_available"] & (dev["durations"] <= 30)
            )
            y_fit, y_select = y[train_mask], y_dev[dev_mask]
            if set(y_fit) != set(range(7)):
                raise ValueError(
                    f"All seven classes must have eligible {modality} training examples"
                )
            scaler = StandardScaler().fit(modality_features(train, modality)[train_mask])
            x = scaler.transform(modality_features(train, modality)[train_mask])
            x_dev = scaler.transform(modality_features(dev, modality)[dev_mask])
            best_score = -1
            trials = []
            for weight in [None, "balanced"]:
                for c in [0.001, 0.01, 0.1, 1.0]:
                    classifier = LogisticRegression(
                        C=c, class_weight=weight, max_iter=1800, random_state=42
                    ).fit(x, y_fit)
                    logits = classifier.decision_function(x_dev)
                    score = f1_score(
                        y_select,
                        logits.argmax(1),
                        average="macro",
                        labels=range(7),
                        zero_division=0,
                    )
                    trials.append(
                        {
                            "C": c,
                            "class_weight": weight,
                            "dev_macro_f1": score,
                            "iterations": classifier.n_iter_.tolist(),
                        }
                    )
                    print(modality, trials[-1], flush=True)
                    if score > best_score:
                        best_score, best, best_logits = score, classifier, logits
            temperature = float(
                minimize_scalar(
                    lambda t, logits=best_logits, target=y_select: log_loss(
                        target, softmax(logits / t, axis=1), labels=range(7)
                    ),
                    bounds=(0.2, 5.0),
                    method="bounded",
                ).x
            )
            for name, value in {
                "mean": scaler.mean_,
                "scale": scaler.scale_,
                "coef": best.coef_,
                "intercept": best.intercept_,
                "temperature": np.array(temperature),
            }.items():
                weights[f"{modality}_{name}"] = np.asarray(value, dtype=np.float64)
            results[modality] = metrics(y_select, softmax(best_logits / temperature, axis=1))
            selected[modality] = {
                "C": best.C,
                "class_weight": best.class_weight,
                "temperature": temperature,
                "trials": trials,
                "training_rows": int(train_mask.sum()),
                "development_rows": int(dev_mask.sum()),
            }
    common = dev["audio_available"] & (dev["durations"] <= 30)
    calibrated = {}
    for name in ("text", "fusion"):
        x = modality_features(dev, name)[common]
        z = (x - weights[f"{name}_mean"]) / weights[f"{name}_scale"]
        calibrated[name] = softmax(
            (z @ weights[f"{name}_coef"].T + weights[f"{name}_intercept"])
            / weights[f"{name}_temperature"],
            axis=1,
        )
    mix_trials = []
    for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
        p = alpha * calibrated["fusion"] + (1 - alpha) * calibrated["text"]
        mix_trials.append(
            {
                "fusion_weight": alpha,
                "dev_macro_f1": f1_score(
                    y_dev[common], p.argmax(1), average="macro", labels=range(7), zero_division=0
                ),
            }
        )
    # Alpha zero is reported as a reference; a multimodal system must use audio.
    chosen_mix = max(mix_trials[1:], key=lambda row: row["dev_macro_f1"])
    alpha = chosen_mix["fusion_weight"]
    weights["fusion_mix"] = np.array(alpha)
    selected["probability_mix"] = {
        "chosen_fusion_weight": alpha,
        "trials": mix_trials,
        "selection_rows": int(common.sum()),
    }
    results["fusion_head"] = results["fusion"]
    results["text_common_audio"] = metrics(y_dev[common], calibrated["text"])
    results["fusion"] = metrics(
        y_dev[common], alpha * calibrated["fusion"] + (1 - alpha) * calibrated["text"]
    )
    print("Selected probability mixture:", selected["probability_mix"], flush=True)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "heads.npz", **weights)
    meta = {
        "format_version": 1,
        "labels": LABELS,
        "text_revision": TEXT_REVISION,
        "audio_revision": AUDIO_REVISION,
        "training_rows": len(y),
        "dev_rows": len(y_dev),
        "seed": 42,
        "selection": selected,
        "head_and_normalization_parameters": sum(v.size for v in weights.values()),
        "response_policy_parameters": 0,
        "uncertainty_threshold": 0.45,
    }
    (output / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    Path(report).parent.mkdir(parents=True, exist_ok=True)
    Path(report).write_text(
        json.dumps({"split": "dev", "metrics": results, "selection": selected}, indent=2) + "\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True)
    parser.add_argument("--dev", required=True)
    parser.add_argument("--output", default="artifacts/model")
    parser.add_argument("--report", default="reports/development.json")
    args = parser.parse_args()
    fit(args.train, args.dev, args.output, args.report)


if __name__ == "__main__":
    main()
