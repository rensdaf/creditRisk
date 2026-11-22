"""Non-régression : comparaison à une baseline FIGÉE (reports/baseline.json), jamais recalculée ici."""
import json
from pathlib import Path

import numpy as np

from app.evaluation import TOLERANCES, compare_to_baseline, evaluate_model
from app.prediction import predict_default
from app.training import fit_reference, load_bundle, predict_proba_frame, save_bundle
from app.validation import validate_applicant

BASELINE = json.loads((Path(__file__).resolve().parents[1] / "reports" / "baseline.json").read_text())


def _validation_metrics(ref):
    _, X_va, _, _, y_va, _ = ref["splits"]
    proba = predict_proba_frame(ref["bundle"], X_va)
    pred = (proba >= BASELINE["seuil_decision"]).astype(int)  # seuil GELÉ de la baseline
    return evaluate_model(y_va, pred, proba)


def test_baseline_file_is_complete():
    for key in ["code_version", "bibliotheques", "graine", "seuil_decision", "dataset", "metrics_validation",
                "profils_reference", "tolerance_probabilite", "contrat_reponse_predict"]:
        assert key in BASELINE


def test_tolerances_are_defined_before_comparison():
    assert set(TOLERANCES) == {"recall", "f1", "roc_auc"} and all(v > 0 for v in TOLERANCES.values())


def test_current_version_within_tolerance(ref):
    checks = compare_to_baseline(_validation_metrics(ref), BASELINE["metrics_validation"])
    assert all(checks.values()), checks


def test_reference_profiles_probabilities_stable(bundle):
    tol = BASELINE["tolerance_probabilite"]
    for profil in BASELINE["profils_reference"]:
        p = predict_default(bundle, validate_applicant(profil["entree"]))["probabilite_defaut"]
        assert abs(p - profil["probabilite"]) <= tol, profil["nom"]


def test_response_contract_unchanged(bundle, payload):
    assert sorted(predict_default(bundle, validate_applicant(payload))) == BASELINE["contrat_reponse_predict"]


def test_degraded_model_is_detected():
    """Contrôle du contrôle : un modèle entraîné sur des étiquettes mélangées doit être rejeté."""
    degraded = fit_reference(shuffle_labels=True)
    checks = compare_to_baseline(_validation_metrics(degraded), BASELINE["metrics_validation"])
    assert not all(checks.values())


def test_save_and_reload_predictions_identical(tmp_path, ref):
    path = tmp_path / "m.joblib"
    save_bundle(ref["bundle"], path)
    reloaded = load_bundle(path)
    X_te = ref["splits"][2]
    np.testing.assert_allclose(predict_proba_frame(ref["bundle"], X_te), predict_proba_frame(reloaded, X_te),
                               atol=1e-12)


def test_training_is_reproducible(ref):
    again = fit_reference()
    X_te = ref["splits"][2]
    np.testing.assert_allclose(predict_proba_frame(ref["bundle"], X_te), predict_proba_frame(again["bundle"], X_te),
                               atol=1e-10)
