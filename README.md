# Prévision de consommation électrique à 24 h

Prédire la consommation électrique **horaire** d'une maison sur les **24 prochaines heures**, à partir de l'historique, du calendrier et de la météo.

---

## Contenu du dépôt

| Fichier | Rôle |
|---|---|
| `Dataset_Prevision_de_Consommation_Electrique.ipynb` | Construit le dataset horaire et l'enregistre sur Google Drive |
| `LSTM_Prevision_de_Consommation_Electrique.ipynb` | Entraîne et évalue le modèle LSTM |

## Utilisation (Google Colab)

1. Lancer le **notebook Dataset** : il crée `dataset_hourly.parquet` dans `MyDrive/projet_conso/data/processed/`.
2. Lancer un **notebook de modèle** avec le GPU activé : il lit le dataset sur Drive, entraîne le modèle et l'enregistre dans `MyDrive/projet_conso/results/`.

Les données et les modèles ne sont pas dans le dépôt : ils sont générés par les notebooks.

---

## Données

- **Consommation :** [UCI – Individual Household Electric Power Consumption](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption), une maison à Sceaux, mesures à la minute de déc. 2006 à nov. 2010.
- **Météo :** température ressentie horaire à Sceaux, via [Open-Meteo](https://open-meteo.com/en/docs/historical-weather-api).
- **Jours fériés :** librairie `holidays` (France).

## Construction du dataset

1. **Nettoyage :** index `datetime`, valeurs `?` converties en manquantes.
2. **Agrégation horaire :** moyenne des minutes (somme pour les sous-compteurs) ; une heure est valide si au moins 45 minutes sont mesurées.
3. **Valeurs manquantes (1,26 %) :** interpolation pour les trous de 3 h ou moins, sinon même heure la semaine précédente. Ces heures sont marquées `is_imputed` et exclues de l'entraînement et de l'évaluation.
4. **Variables ajoutées :** consommation non mesurée (`unmetered_wh`), heure, jour, mois, week-end, jour férié.
5. **Météo :** alignée en heure fixe UTC+1, comme les données UCI.

**Résultat :** `dataset_hourly.parquet`, **34 587 heures × 15 colonnes**, sans valeur manquante. Cible : `global_active_power` (kW).

---

## Protocole d'évaluation

- **Découpage :** train 2007–2008, validation 2009, test 2010.
- **Entrées :** passé (consommation, température, calendrier) + calendrier des 24 heures à prédire.
- **Réglage** sur la validation (3 seeds) ; **test utilisé une seule fois**.
- **Métriques :** MAE, RMSE, WAPE.

## Résultats (test 2010)

| Modèle | MAE (kW) | RMSE (kW) | WAPE (%) |
|---|---|---|---|
| Persistance J-1 | 0,558 | 0,823 | 52,6 |
| Persistance J-7 | 0,576 | 0,831 | 54,2 |
| Profil moyen (jour × heure) | 0,512 | 0,687 | 48,3 |
| **LSTM** | **0,455** | **0,644** | **42,9** |

**LSTM :** encodeur LSTM (2 couches, 64 neurones) sur les 14 derniers jours, puis une tête dense qui prédit les 24 heures d'un coup. Il fait **18 % de mieux** que la persistance J-1. Donner la vraie température future n'améliore pas le résultat.

---

## Avancement

- [x] Dataset
- [x] Baselines simples
- [x] LSTM
- [ ] MLP
- [ ] CNN
- [ ] SARIMA
- [ ] Analyse des erreurs

## Limites

Une seule maison, donc une série très bruitée (four, lessive) qui limite la précision atteignable.
