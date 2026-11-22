import pytest

from app.training import fit_reference

GOOD = {"age": 35, "revenu_mensuel": 250000, "montant_credit": 1000000,
        "duree_mois": 12, "anciennete_activite": 4}
JSON = {"Content-Type": "application/json"}


@pytest.fixture(scope="session")
def ref():
    return fit_reference()


@pytest.fixture(scope="session")
def bundle(ref):
    return ref["bundle"]


@pytest.fixture()
def payload():
    return dict(GOOD)


def _client(app):
    from fastapi.testclient import TestClient
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture()
def client(bundle):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from app.main import create_app
    return _client(create_app(bundle=bundle))


@pytest.fixture()
def client_no_model(tmp_path):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from app.main import create_app
    return _client(create_app(model_path=str(tmp_path / "absent.joblib")))


@pytest.fixture()
def make_client():
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from app.main import create_app

    def factory(**kwargs):
        return _client(create_app(**kwargs))
    return factory
