import math

import numpy as np
import pandas as pd
import pytest

from app.preprocessing import MODEL_FEATURES, RAW_FEATURES, PreprocessingError, build_features, payload_to_frame


def test_features_values_and_dimensions(payload):
    X = payload_to_frame(payload)
    assert list(X.columns) == MODEL_FEATURES and X.shape == (1, 7)
    assert math.isclose(X["ratio_mensualite_revenu"].iloc[0], (1_000_000 / 12) / 250_000)
    assert math.isclose(X["log_revenu"].iloc[0], math.log(250_000))
    assert np.isfinite(X.to_numpy()).all()


def test_column_order_is_enforced(payload):
    shuffled = pd.DataFrame([payload])[list(reversed(RAW_FEATURES))]
    assert list(build_features(shuffled).columns) == MODEL_FEATURES


def test_missing_column_rejected(payload):
    with pytest.raises(PreprocessingError, match="manquantes"):
        build_features(pd.DataFrame([payload]).drop(columns=["age"]))


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_non_finite_rejected(payload, bad):
    with pytest.raises(PreprocessingError):
        build_features(pd.DataFrame([{**payload, "montant_credit": bad}]))


def test_empty_and_wrong_type_rejected(payload):
    with pytest.raises(PreprocessingError):
        build_features(pd.DataFrame(columns=RAW_FEATURES))
    with pytest.raises(PreprocessingError):
        build_features([payload])


def test_zero_income_or_duration_rejected(payload):
    with pytest.raises(PreprocessingError):
        build_features(pd.DataFrame([{**payload, "revenu_mensuel": 0}]))
    with pytest.raises(PreprocessingError):
        build_features(pd.DataFrame([{**payload, "duree_mois": 0}]))


def test_many_rows_keep_row_count(ref):
    X = ref["splits"][1]
    assert len(build_features(X)) == len(X)
