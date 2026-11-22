"""Tests fonctionnels de l'API : codes HTTP, contenu des réponses, comportement applicatif."""
import json

import pytest

from tests.conftest import JSON


def post(client, body):
    return client.post("/predict", json=body)


def test_valid_request(client, payload):  # EX04
    r = post(client, payload)
    assert r.status_code == 200
    data = r.json()
    assert 0 <= data["probabilite_defaut"] <= 1
    assert data["niveau_risque"] in {"faible", "modere", "eleve"}
    assert data["version_modele"] and isinstance(data["alertes"], list)
    assert "ne remplace pas" in data["avertissement"]


def test_response_contract_keys(client, payload):
    assert set(post(client, payload).json()) == {"probabilite_defaut", "niveau_risque", "alertes",
                                                 "version_modele", "avertissement"}


@pytest.mark.parametrize("field", ["age", "revenu_mensuel", "montant_credit", "duree_mois", "anciennete_activite"])
def test_missing_field_returns_422(client, payload, field):  # EX06
    del payload[field]
    r = post(client, payload)
    assert r.status_code == 422
    body = r.json()
    assert body["erreur"] == "donnees_invalides"
    assert any(d["champ"] == field and d["code"] == "champ_manquant" for d in body["details"])


@pytest.mark.parametrize("field,value,code", [
    ("age", -5, "hors_limites"),                   # EX01
    ("revenu_mensuel", "beaucoup", "type_invalide"),  # EX02
    ("montant_credit", 0, "hors_limites"),
    ("montant_credit", -1000, "hors_limites"),
    ("duree_mois", 12.5, "type_invalide"),         # EX03
    ("revenu_mensuel", 1e15, "hors_limites"),      # valeurs extrêmes
    ("montant_credit", 10**30, "hors_limites"),
])
def test_invalid_values_return_422(client, payload, field, value, code):
    r = post(client, {**payload, field: value})
    assert r.status_code == 422
    assert any(d["champ"] == field and d["code"] == code for d in r.json()["details"])


def test_extra_field_returns_422(client, payload):
    r = post(client, {**payload, "nom": "Awa"})
    assert r.status_code == 422
    assert any(d["code"] == "champ_non_autorise" for d in r.json()["details"])


@pytest.mark.parametrize("content", [b"{pas du json", b"", b'{"age": 35,'])
def test_malformed_body_returns_400(client, content):
    r = client.post("/predict", content=content, headers=JSON)
    assert r.status_code == 400 and r.json()["erreur"] == "json_invalide"


def test_non_object_json_returns_422(client):
    r = client.post("/predict", content=b"[1, 2, 3]", headers=JSON)
    assert r.status_code == 422


def test_error_response_does_not_echo_values(client, payload):
    r = post(client, {**payload, "revenu_mensuel": "VALEUR_SENSIBLE_987"})
    assert "VALEUR_SENSIBLE_987" not in r.text and "Traceback" not in r.text


def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json() == {"status": "ok", "modele_charge": True}


def test_health_when_model_unavailable(client_no_model):  # EX05
    r = client_no_model.get("/health")
    assert r.status_code == 503 and r.json()["status"] == "indisponible"


def test_predict_when_model_unavailable(client_no_model, payload):  # EX05
    r = post(client_no_model, payload)
    assert r.status_code == 503 and r.json()["erreur"] == "modele_indisponible"
    assert "Traceback" not in r.text


def test_model_info(client, bundle):
    r = client.get("/model-info")
    assert r.status_code == 200
    info = r.json()
    assert info["version_modele"] == bundle["metadata"]["version_modele"]
    assert len(info["variables"]) == 7 and "SYNTHETIQUES" in info["donnees"]
    assert set(info["plages_entrainement"]) >= {"age", "revenu_mensuel"}


def test_model_info_when_model_unavailable(client_no_model):
    assert client_no_model.get("/model-info").status_code == 503


def test_wrong_method_and_unknown_route(client):
    assert client.get("/predict").status_code == 405
    assert client.get("/inconnu").status_code == 404


def test_request_id_present_on_errors(client, payload):
    r = post(client, {**payload, "age": -1})
    assert len(r.json()["id_requete"]) == 12


def test_repeated_requests_are_identical(client, payload):
    assert post(client, payload).json() == post(client, payload).json()
    assert json.dumps(post(client, payload).json())
