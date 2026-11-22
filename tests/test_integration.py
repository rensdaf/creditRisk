"""Intégration validation -> prétraitement -> modèle -> API, et gestion propre des erreurs internes."""
import logging

import numpy as np
import pytest

from app import preprocessing
from app.preprocessing import MODEL_FEATURES, PreprocessingError, payload_to_frame
from app.training import build_pipeline, save_bundle


def test_end_to_end_matches_direct_pipeline(client, bundle, payload):
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    direct = bundle["pipeline"].predict_proba(payload_to_frame(payload))[:, 1][0]
    assert r.json()["probabilite_defaut"] == pytest.approx(direct, abs=1e-12)


def test_metadata_consistent_across_endpoints(client, bundle, payload):
    info = client.get("/model-info").json()
    pred = client.post("/predict", json=payload).json()
    assert info["version_modele"] == pred["version_modele"] == bundle["metadata"]["version_modele"]
    assert info["variables"] == MODEL_FEATURES
    assert info["seuil_decision"] == bundle["metadata"]["seuil_decision"]


def test_save_reload_gives_same_predictions(tmp_path, bundle, make_client, payload):
    path = tmp_path / "model.joblib"
    save_bundle(bundle, path)
    p_mem = make_client(bundle=bundle).post("/predict", json=payload).json()["probabilite_defaut"]
    p_file = make_client(model_path=str(path)).post("/predict", json=payload).json()["probabilite_defaut"]
    assert p_mem == pytest.approx(p_file, abs=1e-12)


@pytest.mark.parametrize("content", [b"pas un modele", b""])
def test_corrupted_model_file_gives_503(tmp_path, make_client, payload, content):
    path = tmp_path / "model.joblib"
    path.write_bytes(content)
    c = make_client(model_path=str(path))
    assert c.get("/health").status_code == 503
    assert c.post("/predict", json=payload).status_code == 503


def test_model_file_with_wrong_structure_gives_503(tmp_path, make_client):
    import joblib
    path = tmp_path / "model.joblib"
    joblib.dump({"autre": 1}, path)
    assert make_client(model_path=str(path)).get("/health").status_code == 503


def test_preprocessing_error_returns_generic_500(client, payload, monkeypatch):
    def boom(_):
        raise PreprocessingError("colonne interne secrète absente")
    monkeypatch.setattr(preprocessing, "payload_to_frame", boom)
    r = client.post("/predict", json=payload)
    assert r.status_code == 500 and r.json()["erreur"] == "erreur_interne"
    assert "secrète" not in r.text and "Traceback" not in r.text and "Preprocessing" not in r.text


def test_dimension_mismatch_returns_generic_500(bundle, make_client, payload):
    rng = np.random.default_rng(0)
    wrong = build_pipeline().fit(rng.normal(size=(60, 3)), [0, 1] * 30)  # 3 variables au lieu de 7
    c = make_client(bundle={"pipeline": wrong, "metadata": bundle["metadata"]})
    r = c.post("/predict", json=payload)
    assert r.status_code == 500 and r.json()["erreur"] == "erreur_interne"
    assert "feature" not in r.text.lower() and "Traceback" not in r.text


def test_internal_prediction_error_is_hidden_and_not_logged_with_data(client, bundle, payload, monkeypatch, caplog):
    def boom(*_args, **_kwargs):
        raise RuntimeError("détail technique confidentiel")
    monkeypatch.setattr(bundle["pipeline"], "predict_proba", boom)
    payload["revenu_mensuel"] = 12345678
    with caplog.at_level(logging.ERROR, logger="creditrisk"):
        r = client.post("/predict", json=payload)
    assert r.status_code == 500 and "confidentiel" not in r.text
    assert "12345678" not in caplog.text and "confidentiel" not in caplog.text
    assert "RuntimeError" in caplog.text


def test_invalid_model_output_returns_500(client, bundle, payload, monkeypatch):
    monkeypatch.setattr(bundle["pipeline"], "predict_proba", lambda X: np.array([[0.0, float("nan")]]))
    assert client.post("/predict", json=payload).status_code == 500

