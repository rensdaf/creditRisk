"""Tests de performance légers (CPU standard). Objectifs justifiés dans reports/rapport_qualification."""
import time
from pathlib import Path

import numpy as np

from app.prediction import predict_default
from app.training import save_bundle
from app.validation import validate_applicant

P95_INFERENCE_MS = 20.0   # inférence + prétraitement d'une demande
P95_API_MS = 200.0        # requête complète (TestClient : sans réseau)
MIN_THROUGHPUT = 20.0     # requêtes/s en séquentiel
MAX_MODEL_KB = 200


def _timings(fn, n=100):
    fn()  # échauffement
    out = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        out.append((time.perf_counter() - t0) * 1000)
    return out


def test_inference_latency_p95(bundle, payload):
    clean = validate_applicant(payload)
    assert np.percentile(_timings(lambda: predict_default(bundle, clean)), 95) < P95_INFERENCE_MS


def test_api_latency_p95_and_throughput(client, payload):
    times = _timings(lambda: client.post("/predict", json=payload))
    assert np.percentile(times, 95) < P95_API_MS
    assert 1000 / np.mean(times) > MIN_THROUGHPUT


def test_model_size(tmp_path, bundle):
    path = tmp_path / "m.joblib"
    save_bundle(bundle, path)
    assert path.stat().st_size / 1024 < MAX_MODEL_KB


def test_no_secret_in_source():
    root = Path(__file__).resolve().parents[1]
    bad = ("AKIA", "BEGIN PRIVATE KEY", "ghp_", "password=")
    for p in list(root.glob("app/*.py")) + list(root.glob("scripts/*.py")):
        assert not any(tok in p.read_text(encoding="utf-8") for tok in bad), p
