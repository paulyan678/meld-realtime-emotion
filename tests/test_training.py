"""Protect held-out data and verify the portable, non-pickle training export."""

import json

import numpy as np
import pytest

from meld_emotion import training
from meld_emotion.constants import AUDIO_REVISION, LABELS, TEXT_REVISION
from meld_emotion.inference import Heads


@pytest.fixture
def classifier_calls(monkeypatch):
    calls = {"fitted": [], "selected": []}

    class CheapClassifier:
        """Replace optimization; exercise real scaling, selection and calibration."""

        def __init__(self, C, class_weight, **kwargs):
            self.C, self.class_weight = C, class_weight

        def fit(self, x, y):
            calls["fitted"].append((x.copy(), y.copy()))
            self.coef_ = np.zeros((7, x.shape[1]))
            self.intercept_ = np.arange(7) * 0.05
            self.n_iter_ = np.array([1])
            return self

        def decision_function(self, x):
            calls["selected"].append(x.copy())
            return x @ self.coef_.T + self.intercept_

    monkeypatch.setattr(training, "LogisticRegression", CheapClassifier)
    return calls


def write_features(path, split, labels=None, offset=0.0):
    labels = np.arange(7) if labels is None else np.asarray(labels)
    n = len(labels)
    text = np.column_stack([np.arange(n), np.arange(n) ** 2]).astype(float) + offset
    audio = np.column_stack([np.arange(n) * 2, np.arange(n) % 2, np.ones(n)]) + offset
    np.savez(
        path,
        ids=np.array([f"{split}/dia0_utt{i}" for i in range(n)]),
        labels=labels,
        text=text,
        audio=audio,
        audio_available=np.ones(n, dtype=bool),
        durations=np.ones(n),
        text_revision=np.array(TEXT_REVISION),
        audio_revision=np.array(AUDIO_REVISION),
        feature_version=np.array("meanstd6s-v1"),
    )
    path.with_suffix(".json").write_text(
        json.dumps(
            {
                "text_revision": TEXT_REVISION,
                "audio_revision": AUDIO_REVISION,
                "rows": n,
            }
        )
    )
    return text, audio


@pytest.mark.parametrize(
    "train_split,dev_split,message",
    [
        ("test", "dev", "only train/"),
        ("train", "test", "only dev/"),
        ("dev", "dev", "only train/"),
    ],
)
def test_test_or_wrong_split_cannot_enter_training(
    tmp_path, monkeypatch, train_split, dev_split, message
):
    train_path, dev_path = tmp_path / "train.npz", tmp_path / "dev.npz"
    write_features(train_path, train_split)
    write_features(dev_path, dev_split)

    def forbidden_fit(*args, **kwargs):
        raise AssertionError("A model must not be fit before split validation")

    monkeypatch.setattr(training, "LogisticRegression", forbidden_fit)
    with pytest.raises(ValueError, match=message):
        training.fit(train_path, dev_path, tmp_path / "model", tmp_path / "report.json")
    assert not (tmp_path / "model").exists()


def test_training_requires_all_seven_classes(tmp_path):
    train_path, dev_path = tmp_path / "train.npz", tmp_path / "dev.npz"
    write_features(train_path, "train", labels=np.arange(6))
    write_features(dev_path, "dev")
    with pytest.raises(ValueError, match="seven classes"):
        training.fit(train_path, dev_path, tmp_path / "model", tmp_path / "report.json")


@pytest.mark.parametrize(
    "key,value",
    [
        ("text_revision", np.array("stale-text-model")),
        ("audio_revision", np.array("stale-audio-model")),
        ("feature_version", np.array("different-pooling")),
        ("text", np.full((7, 2), np.nan)),
        ("audio", np.full((7, 3), np.inf)),
    ],
)
@pytest.mark.parametrize("split", ["train", "dev"])
def test_stale_or_invalid_features_rejected_before_fitting(
    tmp_path, monkeypatch, key, value, split
):
    train_path, dev_path = tmp_path / "train.npz", tmp_path / "dev.npz"
    write_features(train_path, "train")
    write_features(dev_path, "dev")
    path = train_path if split == "train" else dev_path
    with np.load(path, allow_pickle=False) as original:
        data = {key: original[key] for key in original.files}
    data[key] = value
    np.savez(path, **data)

    def forbidden_fit(*args, **kwargs):
        raise AssertionError("A model must not be fit before feature validation")

    monkeypatch.setattr(training, "LogisticRegression", forbidden_fit)
    with pytest.raises(ValueError):
        training.fit(train_path, dev_path, tmp_path / "model", tmp_path / "report.json")


def test_fit_uses_train_statistics_and_exports_all_three_heads(tmp_path, classifier_calls):
    train_path, dev_path = tmp_path / "train.npz", tmp_path / "dev.npz"
    text, audio = write_features(train_path, "train")
    write_features(dev_path, "dev", offset=10000.0)
    output, report_path = tmp_path / "model", tmp_path / "report.json"
    training.fit(train_path, dev_path, output, report_path)
    fitted = classifier_calls["fitted"]
    assert fitted
    for x, y in fitted:
        assert x.shape[0] == len(text)
        np.testing.assert_allclose(x.mean(axis=0), 0, atol=1e-12)
        np.testing.assert_array_equal(y, np.arange(7))

    heads = Heads(output)
    for modality, raw in (
        ("text", text),
        ("audio", audio),
        ("fusion", np.concatenate([text, audio], axis=1)),
    ):
        # The dev offset would move these means if dev data had entered scaler.fit.
        np.testing.assert_allclose(heads.weights[f"{modality}_mean"], raw.mean(axis=0))
        probabilities = heads.probabilities(raw, modality)
        assert probabilities.shape == (len(text), 7)
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
        assert np.isfinite(probabilities).all()
        assert 0.2 <= float(heads.weights[f"{modality}_temperature"]) <= 5.0
    assert heads.metadata["labels"] == LABELS
    assert heads.metadata["training_rows"] == 7
    assert heads.metadata["dev_rows"] == 7
    assert heads.metadata["head_and_normalization_parameters"] == sum(
        value.size for value in heads.weights.values()
    )
    report = json.loads(report_path.read_text())
    assert report["split"] == "dev"
    assert set(report["metrics"]) == {"text", "audio", "fusion", "fusion_head", "text_common_audio"}
    mix = report["selection"]["probability_mix"]
    assert mix["chosen_fusion_weight"] == float(heads.weights["fusion_mix"])
    assert mix["chosen_fusion_weight"] in (0.25, 0.5, 0.75, 1.0)
    assert [trial["fusion_weight"] for trial in mix["trials"]] == [0, 0.25, 0.5, 0.75, 1.0]
    assert mix["selection_rows"] == 7


def test_unavailable_and_overlong_audio_excluded_from_acoustic_fit_and_selection(
    tmp_path, classifier_calls
):
    train_path, dev_path = tmp_path / "train.npz", tmp_path / "dev.npz"
    labels = np.r_[np.arange(7), 0, 1]
    write_features(train_path, "train", labels=labels)
    write_features(dev_path, "dev", labels=labels, offset=10000.0)
    original_features = {}
    for path in (train_path, dev_path):
        with np.load(path, allow_pickle=False) as source:
            data = {key: source[key] for key in source.files}
        data["audio_available"][-2] = False
        data["durations"][-1] = 30.01
        data["durations"][-3] = 30.0  # The inclusive duration boundary stays eligible.
        # Extreme finite placeholders make any accidental scaler contamination visible.
        data["text"][-2:] = 100000.0
        data["audio"][-2:] = -100000.0
        np.savez(path, **data)
        original_features[path] = data
    output, report_path = tmp_path / "model", tmp_path / "report.json"
    training.fit(train_path, dev_path, output, report_path)
    fitted, selected = classifier_calls["fitted"], classifier_calls["selected"]
    assert fitted and selected
    for x, y in fitted:
        expected_labels = labels if x.shape[1] == 2 else labels[:7]
        np.testing.assert_array_equal(y, expected_labels)
        assert len(x) == len(expected_labels)
    for x in selected:
        assert len(x) == (9 if x.shape[1] == 2 else 7)

    heads = Heads(output)
    report = json.loads(report_path.read_text())
    for modality in ("text", "audio", "fusion"):
        raw = training.modality_features(original_features[train_path], modality)
        expected = raw if modality == "text" else raw[:7]
        np.testing.assert_allclose(heads.weights[f"{modality}_mean"], expected.mean(axis=0))
        expected_count = 9 if modality == "text" else 7
        assert report["metrics"][modality]["n"] == expected_count
        assert heads.metadata["selection"][modality]["training_rows"] == expected_count
        assert heads.metadata["selection"][modality]["development_rows"] == expected_count
    assert report["selection"]["probability_mix"]["selection_rows"] == 7
    assert report["metrics"]["text_common_audio"]["n"] == 7


@pytest.mark.parametrize("unavailable", [True, False])
def test_acoustic_filter_cannot_silently_remove_a_training_class(
    tmp_path, classifier_calls, unavailable
):
    train_path, dev_path = tmp_path / "train.npz", tmp_path / "dev.npz"
    write_features(train_path, "train")
    write_features(dev_path, "dev")
    with np.load(train_path, allow_pickle=False) as original:
        data = {key: original[key] for key in original.files}
    if unavailable:
        data["audio_available"][-1] = False
    else:
        data["durations"][-1] = 30.01
    np.savez(train_path, **data)
    with pytest.raises(ValueError, match="seven classes"):
        training.fit(train_path, dev_path, tmp_path / "model", tmp_path / "report.json")
    # Text may already be fitted, but an acoustic model must not lose label alignment.
    assert all(x.shape[1] == 2 for x, _ in classifier_calls["fitted"])


def test_metrics_keep_all_seven_classes_when_some_are_absent():
    probabilities = np.full((2, 7), 0.01)
    probabilities[0, 0] = 0.94
    probabilities[1, 1] = 0.94
    result = training.metrics(np.array([0, 1]), probabilities)
    assert result["n"] == 2
    assert result["accuracy"] == 1.0
    assert result["macro_f1"] == pytest.approx(2 / 7)
    assert result["weighted_f1"] == 1.0
    assert np.asarray(result["confusion_matrix"]).shape == (7, 7)
    assert all(label in result["per_class"] for label in LABELS)


def test_fusion_feature_order_matches_online_concatenation():
    data = {"text": np.array([[10, 20]]), "audio": np.array([[30, 40, 50]])}
    np.testing.assert_array_equal(
        training.modality_features(data, "fusion"), np.array([[10, 20, 30, 40, 50]])
    )
