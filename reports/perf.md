# Performance de l'API

{"os": "Windows-11-10.0.26200-SP0", "python": "3.13.2", "cpu_logiques": 24, "ram_go": 16.8}

Requête isolée : P50 = 4.15 ms, P95 = 25.82 ms, débit séquentiel ≈ 141 req/s

| Clients simultanés | Requêtes | Débit (req/s) | P50 (ms) | P95 (ms) |
|---|---|---|---|---|
| 1 | 100 | 126 | 4.16 | 26.04 |
| 2 | 200 | 241 | 7.53 | 10.57 |
| 4 | 400 | 245 | 16.04 | 20.30 |
| 8 | 800 | 251 | 31.39 | 40.17 |