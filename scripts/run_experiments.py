"""Évaluation du modèle (jeu synthétique) : métriques, matrice de confusion, FNR par segment, latence d'inférence.
Sorties : reports/metrics.json, reports/resultats.md, reports/figures/*.png"""
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay  # noqa: E402

from app.evaluation import evaluate_model, metrics_by_segment  # noqa: E402
from app.prediction import predict_default  # noqa: E402
from app.training import fit_reference, predict_proba_frame  # noqa: E402
from app.validation import validate_applicant  # noqa: E402

OUT = Path("reports")
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

ref = fit_reference()
bundle = ref["bundle"]
X_tr, X_va, X_te, y_tr, y_va, y_te = ref["splits"]
thr = bundle["metadata"]["seuil_decision"]  # choisi sur la VALIDATION

res = {"seuil": thr}
for name, X, y in [("train", X_tr, y_tr), ("validation", X_va, y_va), ("test", X_te, y_te)]:
    p = predict_proba_frame(bundle, X)
    res[name] = evaluate_model(y, (p >= thr).astype(int), p)

p_te = predict_proba_frame(bundle, X_te)
pred_te = (p_te >= thr).astype(int)

# --- figures ---
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
RocCurveDisplay.from_predictions(y_te, p_te, name="régression logistique", ax=ax[0])
PrecisionRecallDisplay.from_predictions(y_te, p_te, name="régression logistique", ax=ax[1])
ax[0].set_title("Courbe ROC (test)")
ax[1].set_title("Courbe précision-rappel (test)")
fig.tight_layout()
fig.savefig(FIG / "roc_pr.png", dpi=130)
fig, ax = plt.subplots(figsize=(4.5, 4))
ConfusionMatrixDisplay.from_predictions(y_te, pred_te, display_labels=["Remboursé", "Défaut"], ax=ax, colorbar=False)
ax.set_title(f"Matrice de confusion (seuil={thr:.3f})")
fig.tight_layout()
fig.savefig(FIG / "confusion.png", dpi=130)

# --- analyse par segments (faux négatifs) ---
age_seg = pd.cut(X_te["age"], [17, 24, 34, 49, 75], labels=["18-24", "25-34", "35-49", "50-75"])
inc_seg = pd.qcut(X_te["revenu_mensuel"], 3, labels=["revenu bas", "revenu moyen", "revenu élevé"])
seg_age = metrics_by_segment(y_te, pred_te, age_seg)
seg_inc = metrics_by_segment(y_te, pred_te, inc_seg)
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
for a, seg, title in [(ax[0], seg_age, "FNR par tranche d'âge"), (ax[1], seg_inc, "FNR par tranche de revenu")]:
    a.bar(seg["segment"], seg["fnr"])
    a.axhline(res["test"]["fnr"], color="red", ls="--", label="FNR global")
    a.set_title(title)
    a.set_ylim(0, 1)
    a.legend()
fig.tight_layout()
fig.savefig(FIG / "fnr_segments.png", dpi=130)

# --- latence d'inférence (prédiction isolée, hors HTTP) ---
clean = validate_applicant({"age": 35, "revenu_mensuel": 250000, "montant_credit": 1000000,
                            "duree_mois": 12, "anciennete_activite": 4})
predict_default(bundle, clean)
lat = []
for _ in range(300):
    t0 = time.perf_counter()
    predict_default(bundle, clean)
    lat.append((time.perf_counter() - t0) * 1000)
res["inference"] = {"p50_ms": float(np.percentile(lat, 50)), "p95_ms": float(np.percentile(lat, 95)),
                    "protocole": "300 prédictions isolées après échauffement, time.perf_counter, hors HTTP"}
res["segments_age"] = seg_age.to_dict("records")
res["segments_revenu"] = seg_inc.to_dict("records")
(OUT / "metrics.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))

keys = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "fpr", "fnr", "tp", "fp", "fn", "tn"]
md = ["# Résultats automatiques (jeu SYNTHÉTIQUE)\n", f"Seuil de décision (choisi sur la validation) : {thr:.3f}\n",
      "| Jeu | " + " | ".join(keys) + " |", "|---|" + "---|" * len(keys)]
for n in ("train", "validation", "test"):
    md.append(f"| {n} | " + " | ".join(f"{res[n][k]:.4f}" if isinstance(res[n][k], float) else str(res[n][k]) for k in keys) + " |")
for title, seg in [("tranche d'âge", seg_age), ("tranche de revenu", seg_inc)]:
    md += [f"\n## Faux négatifs par {title} (test)\n", "| Segment | n | défauts | rappel | FNR | FPR |", "|---|---|---|---|---|---|"]
    md += [f"| {r.segment} | {r.n} | {r.defauts} | {r.recall:.3f} | {r.fnr:.3f} | {r.fpr:.3f} |" for r in seg.itertuples()]
md += ["\n## Latence d'inférence\n", f"P50 = {res['inference']['p50_ms']:.2f} ms, P95 = {res['inference']['p95_ms']:.2f} ms"]
(OUT / "resultats.md").write_text("\n".join(md), encoding="utf-8")
print("\n".join(md))
