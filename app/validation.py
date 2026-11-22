"""Module de validation : contrat d'entrée STRICT du demandeur de crédit.

Politique : toute entrée invalide est REJETÉE avec une erreur contrôlée (aucune correction silencieuse).
Les messages d'erreur ne reproduisent jamais les valeurs reçues (pas de fuite de données dans les réponses).
Les valeurs valides mais inhabituelles ne sont pas rejetées ici : elles déclenchent une alerte (prediction.py).
"""
import math

FIELD_SPECS = {
    "age": {"kind": "int", "min": 18, "max": 75},
    "revenu_mensuel": {"kind": "number", "min": 0, "min_exclusive": True, "max": 50_000_000},
    "montant_credit": {"kind": "number", "min": 0, "min_exclusive": True, "max": 500_000_000},
    "duree_mois": {"kind": "int", "min": 1, "max": 120},
    "anciennete_activite": {"kind": "number", "min": 0, "max": 60},
}
AGE_MIN_ACTIVITE = 14  # l'ancienneté d'activité ne peut pas dépasser (âge - 14)


class ValidationFailure(Exception):
    """Levée quand l'entrée est invalide ; `errors` est une liste de dicts {champ, code, message}."""

    def __init__(self, errors):
        super().__init__("Données invalides")
        self.errors = errors


def _err(champ, code, message):
    return {"champ": str(champ)[:50], "code": code, "message": message}


def _check_field(name, value, spec):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        attendu = "entier" if spec["kind"] == "int" else "nombre"
        return _err(name, "type_invalide", f"Type invalide : un {attendu} est attendu.")
    if spec["kind"] == "int" and not isinstance(value, int):
        return _err(name, "type_invalide", "Type invalide : un entier est attendu.")
    if isinstance(value, float) and not math.isfinite(value):
        return _err(name, "valeur_non_finie", "La valeur doit être un nombre fini.")
    low, high = spec["min"], spec["max"]
    too_low = value <= low if spec.get("min_exclusive") else value < low
    if too_low or value > high:
        borne = f"> {low}" if spec.get("min_exclusive") else f">= {low}"
        return _err(name, "hors_limites", f"Valeur hors limites autorisées ({borne} et <= {high}).")
    return None


def validate_applicant(payload):
    """Retourne un dict nettoyé (int/float natifs) ou lève ValidationFailure."""
    if not isinstance(payload, dict):
        raise ValidationFailure([_err("corps", "format_invalide", "Un objet JSON est attendu.")])
    errors = []
    for key in payload:
        if key not in FIELD_SPECS:
            errors.append(_err(key, "champ_non_autorise", "Champ non autorisé."))
    clean = {}
    for name, spec in FIELD_SPECS.items():
        if name not in payload:
            errors.append(_err(name, "champ_manquant", "Champ obligatoire manquant."))
            continue
        problem = _check_field(name, payload[name], spec)
        if problem:
            errors.append(problem)
        else:
            clean[name] = int(payload[name]) if spec["kind"] == "int" else float(payload[name])
    if not errors and clean["anciennete_activite"] > clean["age"] - AGE_MIN_ACTIVITE:
        errors.append(_err("anciennete_activite", "incoherent",
                           "Ancienneté d'activité incompatible avec l'âge."))
    if errors:
        raise ValidationFailure(errors)
    return clean
