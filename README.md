# CreditRisk-Test — qualification d'un système IA de risque de défaut de crédit

API FastAPI (`/health`, `/model-info`, `/predict`) autour d'un modèle de régression logistique, avec une stratégie de test complète :
validation, tests fonctionnels de l'API, intégration, robustesse, non-régression, performance, CI GitHub Actions.

> Les données sont **synthétiques** (voir `app/synthetic.py`) : elles servent à tester le système et ne démontrent pas la performance réelle d'un modèle de crédit.

## Exécution locale
```bash
pip install -r requirements.txt
pytest --cov=app                                  # toute la suite (le modèle de test est entraîné par les fixtures, graine 42)
ruff check .                                      # style
python scripts/train_model.py                     # crée models/model.joblib pour lancer l'API
uvicorn app.main:app --reload                     # http://127.0.0.1:8000/docs
PYTHONPATH=. python scripts/run_experiments.py    # métriques, FNR par segment, figures -> reports/
PYTHONPATH=. python scripts/perf_report.py        # latence P95, débit, charge progressive de l'API
```
Sous PowerShell : `$env:PYTHONPATH = "."` puis `python scripts/run_experiments.py`.

Exemple de requête :
```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" \
  -d '{"age":35,"revenu_mensuel":250000,"montant_credit":1000000,"duree_mois":12,"anciennete_activite":4}'
```

## Contrat de l'API
| Code | Situation |
|---|---|
| 200 | prédiction : `probabilite_defaut`, `niveau_risque`, `alertes`, `version_modele`, `avertissement` |
| 400 | corps JSON mal formé |
| 422 | données invalides (champ manquant, type, limites, champ non autorisé, incohérence) |
| 503 | modèle indisponible |
| 500 | erreur interne générique (identifiant de requête, aucune trace ni valeur) |

## Structure
`app/` (validation, prétraitement, modèle, prédiction, API) · `tests/` · `scripts/` (baseline, expériences, performance, Locust) · `reports/` (plan de test, rapport de qualification, résultats).
La baseline `reports/baseline.json` est créée **une fois** par `scripts/make_baseline.py` puis versionnée ; elle n'est jamais recalculée par la CI.

## Preuves CI (à compléter)
- Exécution réussie : _URL du run_
- Échec volontaire détecté (Pull Request) : _URL de la PR_
