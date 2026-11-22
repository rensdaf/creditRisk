"""Performance de l'API (TestClient, sans réseau) : latence P50/P95, débit, charge progressive.
Sorties : reports/perf.json et reports/perf.md. Pour une charge réseau réelle : scripts/locustfile.py."""
import json
import os
import platform
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import psutil
from fastapi.testclient import TestClient

from app.main import create_app
from app.training import fit_reference

PAYLOAD = {"age": 35, "revenu_mensuel": 250000, "montant_credit": 1000000, "duree_mois": 12, "anciennete_activite": 4}
app = create_app(bundle=fit_reference()["bundle"])


def one_call(client):
    t0 = time.perf_counter()
    r = client.post("/predict", json=PAYLOAD)
    assert r.status_code == 200
    return (time.perf_counter() - t0) * 1000


def worker(n):
    client = TestClient(app)
    return [one_call(client) for _ in range(n)]


client = TestClient(app)
for _ in range(20):
    one_call(client)
single = [one_call(client) for _ in range(300)]
levels = []
for workers in (1, 2, 4, 8):
    per_worker = 100
    t0 = time.perf_counter()
    with ThreadPoolExecutor(workers) as pool:
        lat = [x for chunk in pool.map(worker, [per_worker] * workers) for x in chunk]
    dt = time.perf_counter() - t0
    levels.append({"clients_simultanes": workers, "requetes": len(lat), "debit_req_s": len(lat) / dt,
                   "p50_ms": float(np.percentile(lat, 50)), "p95_ms": float(np.percentile(lat, 95))})
out = {"configuration": {"os": platform.platform(), "python": platform.python_version(),
                         "cpu_logiques": os.cpu_count(), "ram_go": round(psutil.virtual_memory().total / 1e9, 1)},
       "requete_isolee": {"p50_ms": float(np.percentile(single, 50)), "p95_ms": float(np.percentile(single, 95)),
                          "debit_sequentiel_req_s": 1000 / float(np.mean(single))},
       "charge_progressive": levels,
       "protocole": "TestClient FastAPI en processus (HTTP simulé, sans réseau), 300 requêtes isolées après "
                    "échauffement, puis 1/2/4/8 clients en threads x 100 requêtes"}
Path("reports").mkdir(exist_ok=True)
Path("reports/perf.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
md = ["# Performance de l'API\n", json.dumps(out["configuration"], ensure_ascii=False),
      f"\nRequête isolée : P50 = {out['requete_isolee']['p50_ms']:.2f} ms, P95 = {out['requete_isolee']['p95_ms']:.2f} ms, "
      f"débit séquentiel ≈ {out['requete_isolee']['debit_sequentiel_req_s']:.0f} req/s\n",
      "| Clients simultanés | Requêtes | Débit (req/s) | P50 (ms) | P95 (ms) |", "|---|---|---|---|---|"]
md += [f"| {m['clients_simultanes']} | {m['requetes']} | {m['debit_req_s']:.0f} | {m['p50_ms']:.2f} | {m['p95_ms']:.2f} |"
       for m in levels]
Path("reports/perf.md").write_text("\n".join(md), encoding="utf-8")
print("\n".join(md))
