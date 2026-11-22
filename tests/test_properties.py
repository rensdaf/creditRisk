import pytest

pytest.importorskip("hypothesis")  # installé en CI via requirements.txt
from hypothesis import given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

from app.prediction import predict_default  # noqa: E402
from app.validation import ValidationFailure, validate_applicant  # noqa: E402

json_values = st.recursive(
    st.none() | st.booleans() | st.integers() | st.floats(allow_nan=True, allow_infinity=True) | st.text(max_size=20),
    lambda children: st.lists(children, max_size=3) | st.dictionaries(st.text(max_size=12), children, max_size=6),
    max_leaves=12,
)


@settings(max_examples=200, deadline=None)
@given(payload=json_values)
def test_validation_never_crashes_on_arbitrary_json(payload):
    try:
        clean = validate_applicant(payload)
    except ValidationFailure:
        return
    assert set(clean) == {"age", "revenu_mensuel", "montant_credit", "duree_mois", "anciennete_activite"}


@st.composite
def valid_payloads(draw):
    age = draw(st.integers(18, 75))
    return {
        "age": age,
        "revenu_mensuel": draw(st.floats(0.01, 50_000_000, allow_nan=False)),
        "montant_credit": draw(st.floats(0.01, 500_000_000, allow_nan=False)),
        "duree_mois": draw(st.integers(1, 120)),
        "anciennete_activite": draw(st.floats(0, min(60, age - 14), allow_nan=False)),
    }


@settings(max_examples=100, deadline=None)
@given(payload=valid_payloads())
def test_probability_always_in_unit_interval_for_valid_inputs(bundle, payload):
    res = predict_default(bundle, validate_applicant(payload))
    assert 0.0 <= res["probabilite_defaut"] <= 1.0
