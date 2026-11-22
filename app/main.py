"""API de prédiction (FastAPI) : GET /health, GET /model-info, POST /predict.

Le corps de /predict est lu en JSON brut puis validé par app/validation.py (contrat strict, messages
sans écho des valeurs). Codes HTTP : 200 succès ; 400 JSON mal formé ; 422 données invalides ;
503 modèle indisponible ; 500 erreur interne générique (identifiant de requête, aucune trace technique).
"""
import logging
import os
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app import prediction, validation
from app.training import load_bundle

logger = logging.getLogger("creditrisk")
API_VERSION = "1.0.0"
DEFAULT_MODEL_PATH = os.environ.get("MODEL_PATH", "models/model.joblib")


def _error(status, code, message, request_id=None, details=None):
    body = {"erreur": code, "message": message}
    if details is not None:
        body["details"] = details
    if request_id:
        body["id_requete"] = request_id
    return JSONResponse(status_code=status, content=body)


def create_app(model_path=None, bundle=None):
    app = FastAPI(title="CreditRisk-Test API", version=API_VERSION)
    app.state.bundle = bundle
    app.state.load_error = None
    if bundle is None:
        try:
            app.state.bundle = load_bundle(model_path or DEFAULT_MODEL_PATH)
        except Exception as exc:  # modèle absent/corrompu : le service démarre en mode dégradé
            app.state.load_error = type(exc).__name__
            logger.error("Chargement du modèle impossible : %s", type(exc).__name__)

    @app.get("/health")
    def health():
        if app.state.bundle is None:
            return JSONResponse(status_code=503, content={"status": "indisponible", "modele_charge": False})
        return {"status": "ok", "modele_charge": True}

    @app.get("/model-info")
    def model_info():
        if app.state.bundle is None:
            return _error(503, "modele_indisponible", "Le modèle n'est pas disponible.")
        meta = app.state.bundle["metadata"]
        return {k: meta[k] for k in ("version_modele", "algorithme", "variables", "graine", "seuil_decision",
                                     "seuils_niveau_risque", "plages_entrainement", "donnees")}

    @app.post("/predict")
    async def predict(request: Request):
        request_id = uuid.uuid4().hex[:12]
        bundle_ = app.state.bundle
        if bundle_ is None:
            return _error(503, "modele_indisponible", "Le modèle n'est pas disponible.", request_id)
        try:
            payload = await request.json()
        except Exception:
            return _error(400, "json_invalide", "Corps de requête mal formé (JSON attendu).", request_id)
        try:
            clean = validation.validate_applicant(payload)
        except validation.ValidationFailure as exc:
            return _error(422, "donnees_invalides", "Les données du demandeur sont invalides.",
                          request_id, exc.errors)
        try:
            return prediction.predict_default(bundle_, clean)
        except Exception as exc:  # aucune trace ni valeur dans la réponse ni dans les journaux
            logger.error("Échec de prédiction id=%s type=%s", request_id, type(exc).__name__)
            return _error(500, "erreur_interne", "Une erreur interne est survenue.", request_id)

    return app


app = create_app()
