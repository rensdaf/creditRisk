"""Crée la baseline FIGÉE reports/baseline.json. À lancer UNE fois puis à versionner.
Ne jamais l'exécuter automatiquement en CI : cela masquerait les régressions."""
import json
import platform
import subprocess

import numpy as np
import pandas as pd
import sklearn

from app.evaluation import TOLERANCES, evaluate_model
from app.prediction import predict_default
from app.training import fit_reference, predict_proba_frame
from app.validation import validate_applicant

PROFILS = [
    ("profil_typique", {"age": 35, "revenu_mensuel": 250000, "montant_credit": 1000000, "duree_mois": 12, "anciennete_activite": 4}),
    ("profil_risque_faible", {"age": 50, "revenu_mensuel": 500000, "montant_credit": 500000, "duree_mois": 24, "anciennete_activite": 15}),
    ("profil_risque_eleve", {"age": 22, "revenu_mensuel": 100000, "montant_credit": 2000000, "duree_mois": 6, "anciennete_activite": 1}),
    ("profil_jeune_credit_modeste", {"age": 24, "revenu_mensuel": 180000, "montant_credit": 600000, "duree_mois": 18, "anciennete_activite": 2}),
]


def git_version():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "non-versionné"


ref = fit_reference()
bundle = ref["bundle"]
_, X_va, _, _, y_va, _ = ref["splits"]
thr = bundle["metadata"]["seuil_decision"]
proba = predict_proba_frame(bundle, X_va)
metrics = evaluate_model(y_va, (proba >= thr).astype(int), proba)
profils = [{"nom": n, "entree": e,
            "probabilite": predict_default(bundle, validate_applicant(e))["probabilite_defaut"]} for n, e in PROFILS]
baseline = {
    "code_version": git_version(),
    "modele": bundle["metadata"]["algorithme"],
    "version_modele": bundle["metadata"]["version_modele"],
    "bibliotheques": {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                      "numpy": np.__version__, "pandas": pd.__version__},
    "graine": 42, "seuil_decision": thr,
    "dataset": {"n": 6000, "seed": 42, "synthetique": True},
    "dataset_stats": {"lignes": len(ref["df"]), "taux_defaut": float(ref["df"]["defaut"].mean()),
                      "validation_lignes": len(X_va), "validation_defauts": int(y_va.sum())},
    "metrics_validation": metrics,
    "tolerances": TOLERANCES, "tolerance_probabilite": 0.05,
    "profils_reference": profils,
    "contrat_reponse_predict": sorted(predict_default(bundle, validate_applicant(PROFILS[0][1]))),
}
with open("reports/baseline.json", "w", encoding="utf-8") as f:
    json.dump(baseline, f, indent=2, ensure_ascii=False)
print({k: round(v, 3) for k, v in metrics.items() if isinstance(v, float)}, "seuil", round(thr, 4))
