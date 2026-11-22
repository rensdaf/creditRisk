import joblib
import numpy as np
import pandas as pd
import pytest

from app.evaluation import compare_to_baseline, evaluate_model, metrics_by_segment
from app.training import load_bundle, select_threshold


def test_evaluate_model_known_values():
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_pred = np.array([0, 0, 1, 1, 1, 1, 1, 0])
    proba = np.array([.1, .2, .6, .7, .9, .8, .55, .4])
    m = evaluate_model(y_true, y_pred, proba)
    assert (m["tp"], m["tn"], m["fp"], m["fn"]) == (3, 2, 2, 1)
    assert m["recall"] == pytest.approx(0.75) and m["fnr"] == pytest.approx(0.25)
    assert m["precision"] == pytest.approx(0.6) and m["fpr"] == pytest.approx(0.5)


def test_metrics_by_segment_detects_unequal_false_negative_rates():
    y_true = pd.Series([1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0])
    y_pred = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1])
    seg = pd.Series(pd.Categorical(["A"] * 6 + ["B"] * 6, categories=["A", "B"]))
    out = metrics_by_segment(y_true, y_pred, seg).set_index("segment")
    assert out.loc["A", "fnr"] == 0.0 and out.loc["B", "fnr"] == pytest.approx(0.75)
    assert out.loc["B", "fpr"] == pytest.approx(0.5) and out["n"].sum() == 12


def test_segment_without_positive_gives_nan_recall():
    seg = pd.Series(["A", "A", "B", "B"])
    out = metrics_by_segment(pd.Series([1, 0, 0, 0]), np.array([1, 0, 0, 0]), seg).set_index("segment")
    assert np.isnan(out.loc["B", "recall"])


def test_compare_to_baseline_flags_each_metric():
    ref = {"recall": 0.7, "f1": 0.6, "roc_auc": 0.8}
    assert all(compare_to_baseline({"recall": 0.68, "f1": 0.58, "roc_auc": 0.79}, ref).values())
    flags = compare_to_baseline({"recall": 0.5, "f1": 0.6, "roc_auc": 0.8}, ref)
    assert flags == {"recall": False, "f1": True, "roc_auc": True}


def test_select_threshold_respects_min_recall():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    proba = np.clip(y * 0.3 + rng.random(500) * 0.7, 0, 1)
    t = select_threshold(y, proba, min_recall=0.8)
    assert ((proba >= t) & (y == 1)).sum() / (y == 1).sum() >= 0.8


def test_select_threshold_falls_back_to_f1_when_recall_unreachable():
    y = np.array([0, 0, 0, 1, 1, 0, 1, 0])
    proba = np.array([.9, .8, .7, .1, .2, .6, .3, .5])
    assert 0.0 <= select_threshold(y, proba, min_recall=1.1) <= 1.0


def test_load_bundle_rejects_invalid_file(tmp_path):
    path = tmp_path / "x.joblib"
    joblib.dump({"autre": 1}, path)
    with pytest.raises(ValueError, match="invalide"):
        load_bundle(path)
