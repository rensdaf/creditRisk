"""Entraînement, sélection du seuil, sauvegarde/chargement du modèle (bundle = pipeline + métadonnées)."""
import platform

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.preprocessing import MODEL_FEATURES, RAW_FEATURES, build_features
from app.synthetic import make_credit_df

TARGET = "defaut"
MODEL_VERSION = "1.0.0"
RISK_THRESHOLDS = {"modere": 0.30, "eleve": 0.60}
REQUIRED_BUNDLE_KEYS = {"pipeline", "metadata"}


def build_pipeline(seed=42):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=seed)),
    ])


def split_data(df, seed=42, val_size=0.2, test_size=0.2):
    """Partition stratifiée train/validation/test AVANT toute transformation apprise."""
    X, y = df[RAW_FEATURES], df[TARGET]
    X_tmp, X_te, y_tmp, y_te = train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)
    X_tr, X_va, y_tr, y_va = train_test_split(X_tmp, y_tmp, test_size=val_size / (1 - test_size),
                                              stratify=y_tmp, random_state=seed)
    return X_tr, X_va, X_te, y_tr, y_va, y_te


def select_threshold(y_val, proba_val, min_recall=0.70):
    """Seuil de décision choisi sur la VALIDATION : précision maximale sous contrainte de rappel."""
    precision, recall, thresholds = precision_recall_curve(y_val, proba_val)
    precision, recall = precision[:-1], recall[:-1]
    ok = recall >= min_recall
    if ok.any():
        idx = np.where(ok)[0][np.argmax(precision[ok])]
    else:
        f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
        idx = int(np.argmax(f1))
    return float(thresholds[idx])


def make_bundle(pipeline, threshold, X_train, seed=42, version=MODEL_VERSION):
    ranges = {c: [float(X_train[c].quantile(0.01)), float(X_train[c].quantile(0.99))] for c in RAW_FEATURES}
    return {"pipeline": pipeline, "metadata": {
        "version_modele": version, "algorithme": "LogisticRegression (StandardScaler + class_weight=balanced)",
        "variables": list(MODEL_FEATURES), "graine": seed, "seuil_decision": float(threshold),
        "seuils_niveau_risque": dict(RISK_THRESHOLDS), "plages_entrainement": ranges,
        "donnees": "SYNTHETIQUES : usage de test uniquement, ne démontre pas une performance réelle",
        "bibliotheques": {"python": platform.python_version(), "scikit-learn": sklearn.__version__},
    }}


def fit_reference(n=6000, seed=42, min_recall=0.70, shuffle_labels=False):
    """Entraîne le modèle de référence (déterministe). `shuffle_labels=True` fabrique un modèle dégradé."""
    df = make_credit_df(n, seed)
    X_tr, X_va, X_te, y_tr, y_va, y_te = split_data(df, seed)
    y_fit = pd.Series(np.random.default_rng(seed).permutation(y_tr.to_numpy()), index=y_tr.index) \
        if shuffle_labels else y_tr
    pipe = build_pipeline(seed).fit(build_features(X_tr), y_fit)
    thr = select_threshold(y_va, pipe.predict_proba(build_features(X_va))[:, 1], min_recall)
    return {"df": df, "bundle": make_bundle(pipe, thr, X_tr, seed),
            "splits": (X_tr, X_va, X_te, y_tr, y_va, y_te)}


def predict_proba_frame(bundle, raw_df):
    return bundle["pipeline"].predict_proba(build_features(raw_df))[:, 1]


def save_bundle(bundle, path):
    joblib.dump(bundle, path)


def load_bundle(path):
    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or not REQUIRED_BUNDLE_KEYS <= set(bundle):
        raise ValueError("Fichier de modèle invalide")
    return bundle
