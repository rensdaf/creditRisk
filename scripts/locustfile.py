"""Charge réseau (optionnel) : pip install locust ; uvicorn app.main:app ; locust -f scripts/locustfile.py --host http://127.0.0.1:8000"""
from locust import HttpUser, between, task


class Demandeur(HttpUser):
    wait_time = between(0.1, 0.5)

    @task
    def predict(self):
        self.client.post("/predict", json={"age": 35, "revenu_mensuel": 250000, "montant_credit": 1000000,
                                           "duree_mois": 12, "anciennete_activite": 4})
