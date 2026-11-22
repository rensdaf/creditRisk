"""Robustesse : entrées INVALIDES (rejetées) vs entrées INHABITUELLES mais VALIDES (traitées + alerte)."""
import math

import pytest

from app.prediction import predict_default
from app.validation import validate_applicant
from tests.conftest import JSON


def raw_post(client, text):
    return client.post("/predict", content=text, headers=JSON)


@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity"])
def test_non_finite_json_tokens_rejected(client, token):
    body = ('{"age": 35, "revenu_mensuel": %s, "montant_credit": 1000000, '
            '"duree_mois": 12, "anciennete_activite": 4}' % token)
    r = raw_post(client, body)
    assert r.status_code == 422
    assert any(d["code"] == "valeur_non_finie" for d in r.json()["details"])


@pytest.mark.parametrize("field", ["age", "revenu_mensuel", "montant_credit", "duree_mois", "anciennete_activite"])
@pytest.mark.parametrize("value", ["abc", None, [], {}, True])
def test_non_numeric_values_rejected(client, payload, field, value):
    r = client.post("/predict", json={**payload, field: value})
    assert r.status_code == 422


@pytest.mark.parametrize("override", [
    {"age": 18, "anciennete_activite": 2}, {"age": 75}, {"duree_mois": 1}, {"duree_mois": 120},
    {"revenu_mensuel": 0.01}, {"montant_credit": 0.01}, {"anciennete_activite": 0},
    {"revenu_mensuel": 50_000_000}, {"montant_credit": 500_000_000},
])
def test_boundary_values_accepted_with_valid_probability(client, payload, override):
    r = client.post("/predict", json={**payload, **override})
    assert r.status_code == 200
    p = r.json()["probabilite_defaut"]
    assert math.isfinite(p) and 0.0 <= p <= 1.0


def test_unusual_but_valid_input_is_processed_with_alert(client, payload):
    r = client.post("/predict", json={**payload, "revenu_mensuel": 5_000_000})
    assert r.status_code == 200
    assert "valeur_hors_domaine_entrainement:revenu_mensuel" in r.json()["alertes"]


def test_typical_input_has_no_alert(client, payload):
    assert client.post("/predict", json=payload).json()["alertes"] == []


def test_small_perturbation_changes_probability_only_slightly(bundle, payload):
    base = predict_default(bundle, validate_applicant(payload))["probabilite_defaut"]
    pert = predict_default(bundle, validate_applicant({**payload, "revenu_mensuel": 252_500}))["probabilite_defaut"]
    assert abs(pert - base) < 0.05  # sensibilité mesurée (1 % sur le revenu), pas d'invariance exigée


@pytest.mark.parametrize("field,low,high,direction", [
    ("montant_credit", 500_000, 2_000_000, +1),
    ("duree_mois", 6, 36, -1),
    ("revenu_mensuel", 100_000, 500_000, -1),
    ("age", 22, 50, -1),
    ("anciennete_activite", 0, 10, -1),
])
def test_business_monotonicity_in_typical_domain(bundle, payload, field, low, high, direction):
    p_low = predict_default(bundle, validate_applicant({**payload, field: low}))["probabilite_defaut"]
    p_high = predict_default(bundle, validate_applicant({**payload, field: high}))["probabilite_defaut"]
    assert (p_high - p_low) * direction > 0


@pytest.mark.xfail(strict=True, reason="Anomalie connue A-R1 : extrapolation linéaire hors domaine d'entraînement "
                                       "(un revenu très élevé augmente le risque). À corriger puis retirer ce xfail.")
def test_far_out_of_domain_income_should_not_increase_risk(bundle, payload):
    typical = predict_default(bundle, validate_applicant(payload))["probabilite_defaut"]
    extreme = predict_default(bundle, validate_applicant({**payload, "revenu_mensuel": 49_000_000}))
    assert extreme["probabilite_defaut"] <= typical


def test_far_out_of_domain_input_is_at_least_flagged(bundle, payload):
    res = predict_default(bundle, validate_applicant({**payload, "revenu_mensuel": 49_000_000}))
    assert 0.0 <= res["probabilite_defaut"] <= 1.0
    assert "valeur_hors_domaine_entrainement:revenu_mensuel" in res["alertes"]
