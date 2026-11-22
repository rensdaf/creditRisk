"""Module de prétraitement : construction des variables du modèle à partir des champs validés."""
import numpy as np
import pandas as pd

RAW_FEATURES = ["age", "revenu_mensuel", "montant_credit", "duree_mois", "anciennete_activite"]
MODEL_FEATURES = RAW_FEATURES + ["ratio_mensualite_revenu", "log_revenu"]


class PreprocessingError(ValueError):
    """Erreur de prétraitement (colonnes absentes, valeurs non finies, données vides)."""


def build_features(df):
    """Ajoute le ratio mensualité/revenu et le log du revenu ; retourne les colonnes MODEL_FEATURES."""
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise PreprocessingError("Données vides ou de type invalide")
    missing = [c for c in RAW_FEATURES if c not in df.columns]
    if missing:
        raise PreprocessingError(f"Colonnes manquantes : {missing}")
    out = df[RAW_FEATURES].astype(float).copy()
    if not np.isfinite(out.to_numpy()).all():
        raise PreprocessingError("Valeurs non finies en entrée")
    if (out["revenu_mensuel"] <= 0).any() or (out["duree_mois"] <= 0).any():
        raise PreprocessingError("Revenu et durée doivent être strictement positifs")
    out["ratio_mensualite_revenu"] = (out["montant_credit"] / out["duree_mois"]) / out["revenu_mensuel"]
    out["log_revenu"] = np.log(out["revenu_mensuel"])
    result = out[MODEL_FEATURES]
    if not np.isfinite(result.to_numpy()).all():
        raise PreprocessingError("Valeurs non finies après prétraitement")
    return result


def payload_to_frame(clean):
    """Transforme un dict validé en DataFrame de variables prêtes pour le modèle (1 ligne)."""
    return build_features(pd.DataFrame([clean]))
