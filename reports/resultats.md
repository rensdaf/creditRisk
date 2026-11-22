# Résultats automatiques (jeu SYNTHÉTIQUE)

Seuil de décision (choisi sur la validation) : 0.513

| Jeu | accuracy | precision | recall | f1 | roc_auc | pr_auc | fpr | fnr | tp | fp | fn | tn |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| train | 0.7922 | 0.5251 | 0.6424 | 0.5779 | 0.8210 | 0.6223 | 0.1652 | 0.3576 | 512 | 463 | 285 | 2340 |
| validation | 0.8075 | 0.5503 | 0.7019 | 0.6169 | 0.8295 | 0.6228 | 0.1626 | 0.2981 | 186 | 152 | 79 | 783 |
| test | 0.7692 | 0.4830 | 0.6415 | 0.5511 | 0.7970 | 0.5928 | 0.1947 | 0.3585 | 170 | 182 | 95 | 753 |

## Faux négatifs par tranche d'âge (test)

| Segment | n | défauts | rappel | FNR | FPR |
|---|---|---|---|---|---|
| 18-24 | 126 | 29 | 0.483 | 0.517 | 0.433 |
| 25-34 | 314 | 85 | 0.706 | 0.294 | 0.236 |
| 35-49 | 601 | 129 | 0.643 | 0.357 | 0.148 |
| 50-75 | 159 | 22 | 0.591 | 0.409 | 0.117 |

## Faux négatifs par tranche de revenu (test)

| Segment | n | défauts | rappel | FNR | FPR |
|---|---|---|---|---|---|
| revenu bas | 400 | 76 | 0.592 | 0.408 | 0.170 |
| revenu moyen | 400 | 93 | 0.645 | 0.355 | 0.195 |
| revenu élevé | 400 | 96 | 0.677 | 0.323 | 0.220 |

## Latence d'inférence

P50 = 2.09 ms, P95 = 2.82 ms