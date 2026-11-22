"""Jeu de données SYNTHÉTIQUE documenté (microfinance fictive).

Il sert uniquement à tester le système : il ne permet PAS de démontrer la performance réelle d'un modèle de crédit.
Construction : le risque de défaut augmente avec le ratio mensualité/revenu et diminue avec l'âge et l'ancienneté.
Choix volontaire de test : pour les moins de 25 ans, le lien entre ces variables et le défaut est atténué
(revenus plus volatils), afin de pouvoir démontrer une analyse des faux négatifs par segment.
"""
import numpy as np
import pandas as pd


def make_credit_df(n=6000, seed=42):
    rng = np.random.default_rng(seed)
    age = np.clip(np.round(rng.normal(38, 10, n)), 18, 70).astype(int)
    revenu = np.clip(rng.lognormal(np.log(250_000), 0.5, n), 50_000, 3_000_000).round(-2)
    montant = np.clip((revenu * rng.uniform(1, 8, n)).round(-3), 50_000, 40_000_000)
    duree = rng.choice([6, 12, 18, 24, 36], n)
    anciennete = np.minimum(np.round(rng.gamma(2.0, 2.5, n)), age - 14).astype(int)
    ratio = (montant / duree) / revenu
    signal = 5.0 * ratio - 0.03 * (age - 38) - 0.10 * anciennete
    centered = signal - signal.mean()
    weight = np.where(age < 25, 0.3, 1.0)
    z = -1.6 + weight * centered + rng.normal(0, 0.5, n)
    defaut = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)
    return pd.DataFrame({
        "age": age, "revenu_mensuel": revenu, "montant_credit": montant, "duree_mois": duree,
        "anciennete_activite": anciennete, "defaut": defaut,
    })
