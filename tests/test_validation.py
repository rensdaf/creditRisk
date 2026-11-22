import json

import pytest

from app.validation import ValidationFailure, validate_applicant


def codes(payload):
    with pytest.raises(ValidationFailure) as exc:
        validate_applicant(payload)
    return {(e["champ"], e["code"]) for e in exc.value.errors}


def test_valid_payload_returns_native_types(payload):  # EX01-EX03 cas nominal
    clean = validate_applicant(payload)
    assert clean == payload
    assert isinstance(clean["age"], int) and isinstance(clean["revenu_mensuel"], float)


@pytest.mark.parametrize("age", [18, 75])
def test_age_boundaries_accepted(payload, age):  # EX01 valeurs limites
    payload.update(age=age, anciennete_activite=2)
    assert validate_applicant(payload)["age"] == age


@pytest.mark.parametrize("age", [17, 76, 0, -3, 10**400])
def test_age_out_of_range_rejected(payload, age):  # EX01
    assert ("age", "hors_limites") in codes({**payload, "age": age})


@pytest.mark.parametrize("value", ["250000", None, [250000], True])
def test_revenue_wrong_type_rejected(payload, value):  # EX02
    assert ("revenu_mensuel", "type_invalide") in codes({**payload, "revenu_mensuel": value})


@pytest.mark.parametrize("field", ["revenu_mensuel", "montant_credit"])
@pytest.mark.parametrize("value", [0, -1, -250000.5])
def test_non_positive_amounts_rejected(payload, field, value):  # EX02
    assert (field, "hors_limites") in codes({**payload, field: value})


def test_smallest_positive_amount_accepted(payload):
    assert validate_applicant({**payload, "revenu_mensuel": 0.01})["revenu_mensuel"] == 0.01


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_rejected(payload, value):
    assert ("revenu_mensuel", "valeur_non_finie") in codes({**payload, "revenu_mensuel": value})


@pytest.mark.parametrize("duree", [12.5, 12.0, "12", None, True])
def test_duration_must_be_integer(payload, duree):  # EX03
    assert ("duree_mois", "type_invalide") in codes({**payload, "duree_mois": duree})


@pytest.mark.parametrize("duree", [0, -6, 121])
def test_duration_out_of_range(payload, duree):  # EX03
    assert ("duree_mois", "hors_limites") in codes({**payload, "duree_mois": duree})


@pytest.mark.parametrize("duree", [1, 120])
def test_duration_boundaries_accepted(payload, duree):
    assert validate_applicant({**payload, "duree_mois": duree})["duree_mois"] == duree


@pytest.mark.parametrize("field", ["age", "revenu_mensuel", "montant_credit", "duree_mois", "anciennete_activite"])
def test_missing_field_rejected(payload, field):  # EX06
    del payload[field]
    assert (field, "champ_manquant") in codes(payload)


def test_all_missing_fields_reported_together():
    assert len(codes({})) == 5


def test_extra_field_rejected(payload):
    assert ("statut_marital", "champ_non_autorise") in codes({**payload, "statut_marital": "x"})


@pytest.mark.parametrize("body", [[1, 2], "texte", 12, None])
def test_non_object_payload_rejected(body):
    assert ("corps", "format_invalide") in codes(body)


def test_activity_seniority_consistent_with_age(payload):
    assert ("anciennete_activite", "incoherent") in codes({**payload, "age": 20, "anciennete_activite": 10})


def test_seniority_bounds(payload):
    assert validate_applicant({**payload, "age": 75, "anciennete_activite": 60})
    assert ("anciennete_activite", "hors_limites") in codes({**payload, "age": 75, "anciennete_activite": 61})


def test_error_messages_never_echo_submitted_values(payload):
    secret = "SECRET_VALUE_123"
    with pytest.raises(ValidationFailure) as exc:
        validate_applicant({**payload, "revenu_mensuel": secret, "champ_x": secret})
    assert secret not in json.dumps(exc.value.errors)
