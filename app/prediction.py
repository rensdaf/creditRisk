"""Couche applicative de prédiction : probabilité, niveau de risque, alertes hors domaine."""
import math

from app import preprocessing

AVERTISSEMENT = ("Aide à la décision : cette estimation ne remplace pas l'analyse de l'agent de crédit "
                 "et ne constitue pas une décision d'octroi.")


class PredictionError(RuntimeError):
    """Sortie du modèle invalide (non finie ou hors de [0, 1])."""


def risk_level(proba, thresholds):
    if proba >= thresholds["eleve"]:
        return "eleve"
    if proba >= thresholds["modere"]:
        return "modere"
    return "faible"


def domain_alerts(clean, ranges):
    """Entrées valides mais inhabituelles (hors plage 1 %-99 % d'entraînement) : alerte, pas de rejet."""
    return [f"valeur_hors_domaine_entrainement:{name}" for name, (low, high) in ranges.items()
            if name in clean and not low <= clean[name] <= high]


def predict_default(bundle, clean):
    meta = bundle["metadata"]
    features = preprocessing.payload_to_frame(clean)
    proba = float(bundle["pipeline"].predict_proba(features)[:, 1][0])
    if not math.isfinite(proba) or not 0.0 <= proba <= 1.0:
        raise PredictionError("Probabilité invalide en sortie du modèle")
    return {
        "probabilite_defaut": proba,
        "niveau_risque": risk_level(proba, meta["seuils_niveau_risque"]),
        "alertes": domain_alerts(clean, meta["plages_entrainement"]),
        "version_modele": meta["version_modele"],
        "avertissement": AVERTISSEMENT,
    }
