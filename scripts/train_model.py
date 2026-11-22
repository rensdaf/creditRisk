"""Entraîne le modèle de référence et l'enregistre dans models/model.joblib (à lancer avant uvicorn)."""
from pathlib import Path

from app.training import fit_reference, save_bundle

ref = fit_reference()
Path("models").mkdir(exist_ok=True)
save_bundle(ref["bundle"], "models/model.joblib")
print("Modèle enregistré :", "models/model.joblib", "| seuil =", round(ref["bundle"]["metadata"]["seuil_decision"], 4))
