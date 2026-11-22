# Plan de test et matrice de traçabilité — CreditRisk-Test

## 1. Périmètre
Module de validation (`app/validation.py`), prétraitement (`preprocessing.py`), modèle (régression logistique, `training.py`), couche de prédiction (`prediction.py`) et API FastAPI (`main.py`). Hors périmètre : authentification, déploiement, données réelles de crédit.

## 2. Exigences, cas de test, risques et priorité
Priorité : **P1** = défaillance grave (décision erronée, fuite, indisponibilité non signalée) ; **P2** = défaillance gênante ; **P3** = confort.

| ID | Exigence | Préconditions | Entrées | Résultat attendu | Risque visé | Prio | Tests |
|---|---|---|---|---|---|---|---|
| EX01 | L'âge est dans [18 ; 75] | modèle chargé | âge 17, 18, 75, 76, −3, 10^400 | hors bornes : rejet (422, `hors_limites`) ; bornes : acceptées | demande hors périmètre légal/modèle | P1 | `test_validation`, `test_api` |
| EX02 | Le revenu est numérique et > 0 | idem | texte, 0, négatif, NaN, booléen | erreur contrôlée (422), aucune valeur recopiée | prédiction absurde, injection | P1 | `test_validation`, `test_robustness` |
| EX03 | La durée est un entier de 1 à 120 | idem | 12.5, 12.0, "12", 0, 121 | rejet (422) ; 1 et 120 acceptées | mensualité fausse | P2 | `test_validation`, `test_api` |
| EX04 | La prédiction retourne une probabilité valide | modèle chargé | requête valide, grille de valeurs, propriétés (Hypothesis) | 200 ; 0 ≤ p ≤ 1, finie | score inexploitable | P1 | `test_api`, `test_properties`, `test_robustness` |
| EX05 | Le modèle est disponible, sinon indisponibilité signalée | fichier absent, corrompu ou de structure invalide | GET /health, POST /predict | 503 `modele_indisponible`, sans trace | service dégradé non détecté | P1 | `test_api`, `test_integration` |
| EX06 | Une entrée incomplète est rejetée | modèle chargé | un champ manquant (×5), corps vide | 422 `champ_manquant` | prédiction sur données partielles | P1 | `test_validation`, `test_api` |
| EX07 | Une nouvelle version ne dégrade pas les critères critiques | baseline figée | modèle courant, modèle dégradé | rappel −0,05, F1 −0,05, AUC −0,03 max ; profils de référence à ±0,05 ; dégradation détectée | régression silencieuse | P1 | `test_regression` |
| EX08 | Aucun champ non autorisé n'est accepté | idem | champ supplémentaire (`nom`) | 422 `champ_non_autorise` | fuite/usage de données non prévues | P2 | `test_validation`, `test_api` |
| EX09 | Le montant du crédit est numérique et > 0 | idem | 0, négatif, 1e30 | 422 `hors_limites` | ratio mensualité/revenu faux | P1 | `test_validation`, `test_api` |
| EX10 | L'ancienneté d'activité est cohérente avec l'âge | idem | âge 20, ancienneté 10 | 422 `incoherent` | données contradictoires | P2 | `test_validation` |
| EX11 | Un corps mal formé est refusé proprement | idem | `{pas du json`, corps vide, tableau JSON | 400 ou 422, message générique | plantage, trace exposée | P1 | `test_api` |
| EX12 | Une erreur interne n'expose ni trace ni valeur | erreur provoquée (prétraitement, dimensions, sortie invalide) | requête valide | 500 `erreur_interne` + identifiant ; rien dans la réponse ni les journaux | fuite d'informations sensibles | P1 | `test_integration` |
| EX13 | Une entrée valide mais inhabituelle est traitée avec alerte | modèle chargé | revenu 5 000 000 | 200 + alerte `valeur_hors_domaine_entrainement:revenu_mensuel` | extrapolation du modèle | P2 | `test_robustness` |
| EX14 | Les prédictions sont reproductibles et cohérentes après rechargement | modèle sauvegardé | mêmes entrées avant/après `joblib` | écart ≤ 1e-12 ; entraînement reproductible | modèle déployé ≠ modèle validé | P1 | `test_integration`, `test_regression` |
| EX15 | Performance adaptée à l'usage | machine documentée | 100 à 300 requêtes | P95 inférence < 20 ms ; P95 API < 200 ms ; débit > 20 req/s | attente de l'agent de crédit | P2 | `test_performance`, `perf_report.py` |
| EX16 | Les faux négatifs sont analysés par segment | jeu de test | tranches d'âge et de revenu | FNR par segment publié avec incertitude ; écart au FNR global ≤ 15 points | iniquité, pertes concentrées | P1 | `run_experiments.py`, rapport |

## 3. Scénarios HTTP obligatoires (Partie II)
| # | Scénario | Code attendu | Réponse | Comportement |
|---|---|---|---|---|
| 1 | Requête valide | 200 | probabilité, niveau de risque, alertes, version, avertissement | prédiction produite |
| 2 | Champ obligatoire absent | 422 | `donnees_invalides`, `champ_manquant` | rejet |
| 3 | Âge négatif | 422 | `hors_limites` sur `age` | rejet |
| 4 | Revenu sous forme de texte | 422 | `type_invalide` | rejet, valeur non recopiée |
| 5 | Montant nul ou négatif | 422 | `hors_limites` | rejet |
| 6 | Durée non entière | 422 | `type_invalide` | rejet |
| 7 | Champ supplémentaire | 422 | `champ_non_autorise` | rejet |
| 8 | Valeurs extrêmement élevées | 422 | `hors_limites` | rejet (au-delà des bornes) |
| 9 | Modèle indisponible | 503 | `modele_indisponible` | service dégradé signalé |
| 10 | Requête mal formée | 400 | `json_invalide` | rejet |

## 4. Politique : invalide ≠ inhabituel
Entrée **invalide** (type, borne, champ, cohérence) : rejet explicite. Entrée **valide mais inhabituelle** (hors de la plage 1 %–99 % de l'entraînement) : traitée, avec alerte dans la réponse.

## 5. Outils et preuves
pytest, pytest-cov (`coverage.xml`), Hypothesis, Ruff, FastAPI `TestClient`, joblib, Locust (optionnel), GitHub Actions. Preuves : sortie pytest, couverture, `reports/baseline.json`, `reports/resultats.md`, `reports/perf.md`, figures, URLs des runs CI.

## 6. Critères d'acceptation pour une phase pilote
Ces seuils illustrent la démarche sur un jeu synthétique ; ils devront être validés par le métier sur des données réelles.

| Critère | Acceptation | Rejet |
|---|---|---|
| Tests automatisés | 100 % réussis, couverture ≥ 85 % | tout échec |
| ROC-AUC (test) | ≥ 0,75 | < 0,65 |
| Rappel global (test) | ≥ 0,60 | < 0,50 |
| F1 (test) | ≥ 0,50 | < 0,40 |
| FNR par segment | ≤ FNR global + 15 points, avec incertitude | > + 25 points |
| Entrées invalides | 100 % rejetées | une seule acceptée |
| Non-régression | dans les tolérances | hors tolérance → investigation |
| Latence P95 API | < 200 ms | > 500 ms |

Entre acceptation et rejet : **investiguer**.
