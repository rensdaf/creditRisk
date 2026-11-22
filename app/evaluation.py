"""Métriques, analyse par segments et comparaison à la baseline."""
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)


def evaluate_model(y_true, y_pred, y_proba):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "fpr": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "fnr": float(fn / (fn + tp)) if (fn + tp) else 0.0,
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }


def metrics_by_segment(y_true, y_pred, segments):
    """Rappel / FNR / FPR par segment (Series de même index que y_true)."""
    rows = []
    for name in segments.cat.categories if hasattr(segments, "cat") else sorted(segments.unique()):
        mask = (segments == name).to_numpy()
        yt, yp = np.asarray(y_true)[mask], np.asarray(y_pred)[mask]
        pos, neg = (yt == 1).sum(), (yt == 0).sum()
        tp = ((yt == 1) & (yp == 1)).sum()
        fp = ((yt == 0) & (yp == 1)).sum()
        rows.append({"segment": str(name), "n": int(mask.sum()), "defauts": int(pos),
                     "recall": float(tp / pos) if pos else float("nan"),
                     "fnr": float(1 - tp / pos) if pos else float("nan"),
                     "fpr": float(fp / neg) if neg else float("nan")})
    return pd.DataFrame(rows)


TOLERANCES = {"recall": 0.05, "f1": 0.05, "roc_auc": 0.03}  # baisses maximales admises, fixées AVANT comparaison


def compare_to_baseline(current, reference, tolerances=TOLERANCES):
    """True par métrique si la baisse par rapport à la référence reste dans la tolérance."""
    return {k: (reference[k] - current[k]) <= tol for k, tol in tolerances.items()}
