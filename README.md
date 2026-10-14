# Prévision de consommation électrique à 24 h

Prédire la consommation électrique **horaire** d'une maison sur les **24 prochaines heures**, à partir de l'historique de consommation, du calendrier et de la météo. Le projet compare des baselines, un modèle statistique (SARIMA) et des réseaux de neurones (MLP, CNN, LSTM, TCN).

> **Version actuelle : v0.1 – construction du dataset.**
> Les modèles seront ajoutés au fur et à mesure (voir [Avancement](#9-avancement)).

---

## Sommaire

1. [Problématique](#1-problématique)
2. [Structure du dépôt](#2-structure-du-dépôt)
3. [Installation et exécution](#3-installation-et-exécution)
4. [Données](#4-données)
5. [Construction du dataset](#5-construction-du-dataset)
6. [Dataset final](#6-dataset-final)
7. [Protocole d'évaluation](#7-protocole-dévaluation)
8. [Modèles et résultats](#8-modèles-et-résultats)
9. [Avancement](#9-avancement)
10. [Limites connues](#10-limites-connues)
11. [Références](#11-références)

---

## 1. Problématique

| Élément | Choix |
|---|---|
| Tâche | Régression – prévision de séries temporelles multi-horizon |
| Cible | `global_active_power`, puissance moyenne horaire de la maison (kW = kWh consommés dans l'heure) |
| Horizon | 24 heures, prédites **en une seule fois** (24 sorties) |
| Entrées | Historique (consommation, température, calendrier) + calendrier des 24 heures à prédire |
| Métriques | MAE, RMSE, WAPE (principales), MAPE (secondaire) |

---

## 2. Structure du dépôt

```
.
├── README.md
├── requirements.txt
├── setup_env.sh                 # création de l'environnement virtuel
├── .gitignore
├── scripts/
│   └── download_data.py         # téléchargement automatique des données
├── notebooks/
│   ├── Dataset_Prevision_de_Consommation_Electrique.ipynb   # ✅ construction du dataset
│   └── demo.ipynb               # démonstration (à venir)
├── src/                         # code source (à venir)
│   ├── data_processing.py
│   ├── models.py
│   ├── train.py
│   ├── evaluate.py
│   └── utils.py
├── data/                        # non versionné (voir .gitignore)
│   ├── raw/
│   └── processed/
└── results/                     # modèles, graphiques, métriques
```

---

## 3. Installation et exécution

### Google Colab (recommandé)

1. Ouvrir `notebooks/Dataset_Prevision_de_Consommation_Electrique.ipynb` dans Colab.
2. Exécuter les cellules dans l'ordre. Le notebook monte Google Drive, télécharge les données **une seule fois** (cache Parquet) et écrit le dataset final.

### En local

```bash
git clone <url-du-depot>
cd <nom-du-depot>
bash setup_env.sh            # crée le venv et installe requirements.txt
source .venv/bin/activate
jupyter notebook
```

Dépendances principales : `pandas`, `numpy`, `pyarrow`, `requests`, `holidays`, `ucimlrepo`, `matplotlib`, `torch`.

Les données (`data/`) ne sont pas versionnées : elles sont régénérées par le notebook. Pour forcer un nouveau téléchargement, supprimer le fichier correspondant dans `data/raw/`.

---

## 4. Données

| Source | Contenu | Accès |
|---|---|---|
| [UCI – Individual Household Electric Power Consumption](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption) | 2 075 259 mesures **à la minute** d'une maison à **Sceaux** (7 km de Paris), du 16/12/2006 au 26/11/2010 | `ucimlrepo` (`fetch_ucirepo(id=235)`) |
| [Open-Meteo Historical API](https://open-meteo.com/en/docs/historical-weather-api) | Température ressentie horaire aux coordonnées de Sceaux (48,7786 N ; 2,2906 E) | API publique, sans clé |
| [`holidays`](https://pypi.org/project/holidays/) | Jours fériés officiels en France | `pip install holidays` |

Variables UCI : puissance active et réactive (kW), tension (V), intensité (A), et trois sous-compteurs en Wh : **cuisine** (lave-vaisselle, four, micro-ondes), **buanderie** (lave-linge, sèche-linge, réfrigérateur, éclairage), **chauffe-eau et climatisation**.

**Valeurs manquantes :** 1,25 % des lignes sont vides (codées `?`), sur toutes les colonnes à la fois, alors que les métadonnées UCI indiquent « no missing values ». Tous les horodatages sont présents.

---

## 5. Construction du dataset

```
UCI (1 min, 2 075 259 lignes)
   │ 1. Nettoyage           datetime, '?' -> NaN, renommage des sous-compteurs
   ▼
   │ 2. Agrégation horaire  moyennes / sommes ; heure valide si >= 45 min mesurées
   ▼
   │ 3. Imputation          trous <= 3 h : interpolation ; > 3 h : semaine précédente
   ▼
   │ 4. Variables calculées unmetered_wh + calendrier + jours fériés
   ▼
   │ 5. Fusion météo        Open-Meteo, alignée en UTC+1
   ▼
Dataset final (1 h, 34 587 lignes × 15 colonnes)
```

### Étape 1 – Nettoyage
Fusion de `Date` + `Time` en un index `datetime` trié, conversion numérique (les `?` deviennent `NaN`), renommage des sous-compteurs (`sub_1_kitchen`, `sub_2_laundry`, `sub_3_heater_ac`), suppression des doublons (aucun trouvé).

### Étape 2 – Agrégation horaire
| Variables | Règle | Pourquoi |
|---|---|---|
| Puissances, tension, intensité | Moyenne des minutes | Grandeurs instantanées |
| Sous-compteurs | Moyenne × 60 (= somme), en Wh | Énergies par minute ; la forme « × 60 » reste juste si quelques minutes manquent |

Une heure avec **moins de 45 minutes mesurées** est considérée comme manquante. La première et la dernière heure, incomplètes, sont retirées.

### Étape 3 – Valeurs manquantes
**436 heures manquantes (1,26 %)**, réparties en **14 trous** :

| Taille du trou | Nombre | Méthode | Justification |
|---|---|---|---|
| ≤ 3 h | 7 | Interpolation linéaire | La consommation évolue régulièrement sur quelques heures |
| > 3 h (max. 121 h) | 7 | Même heure, une semaine avant | Conserve l'heure **et** le jour de la semaine |

Chaque heure reconstituée est marquée **`is_imputed = 1`**. Ces heures sont **ignorées dans la perte d'entraînement et dans les métriques**.

### Étape 4 – Variables calculées
- **`unmetered_wh`** = `global_active_power × 1000 − (sub_1 + sub_2 + sub_3)` : la consommation non suivie par les sous-compteurs, soit **51 %** du total en moyenne.
- **Calendrier** : `hour`, `day_of_week`, `month`, `is_weekend`, `is_holiday` (11 jours fériés officiels, soit 42 jours sur la période).

### Étape 5 – Météo
- `apparent_temperature` (température ressentie) de Sceaux, via Open-Meteo.
- Les horodatages UCI **ne suivent pas le changement d'heure** (pas de trou fin mars, pas de doublon fin octobre) : ils sont en heure fixe. La météo est donc téléchargée en **UTC puis décalée à UTC+1**.
- Vérification : en hiver, la température est minimale à **7 h** et maximale à **15 h**, le profil attendu. Après fusion, **0 heure sans météo**.

---

## 6. Dataset final

`data/processed/dataset_hourly.parquet` : **34 587 heures × 15 colonnes**, du 16/12/2006 18:00 au 26/11/2010 20:00, index `datetime` au pas horaire, **0 valeur manquante**.

| # | Colonne | Groupe | Description | Unité | Moyenne |
|---|---|---|---|---|---|
| 1 | `global_active_power` | 🎯 Cible | Consommation moyenne de la maison | kW | 1,09 |
| 2 | `global_reactive_power` | Électrique | Puissance réactive | kW | 0,12 |
| 3 | `voltage` | Électrique | Tension | V | 240,85 |
| 4 | `global_intensity` | Électrique | Courant total | A | 4,62 |
| 5 | `sub_1_kitchen` | Zone | Cuisine | Wh | 67,27 |
| 6 | `sub_2_laundry` | Zone | Buanderie | Wh | 77,73 |
| 7 | `sub_3_heater_ac` | Zone | Chauffe-eau et climatisation | Wh | 387,18 |
| 8 | `unmetered_wh` | Zone | Reste de la maison | Wh | 556,77 |
| 9 | `hour` | Calendrier | Heure (0–23) | – | – |
| 10 | `day_of_week` | Calendrier | Jour (0 = lundi) | – | – |
| 11 | `month` | Calendrier | Mois (1–12) | – | – |
| 12 | `is_weekend` | Calendrier | Samedi ou dimanche | 0/1 | 0,29 |
| 13 | `is_holiday` | Calendrier | Jour férié français | 0/1 | 0,03 |
| 14 | `apparent_temperature` | Météo | Température ressentie | °C | 8,72 |
| 15 | `is_imputed` | Qualité | Heure reconstituée | 0/1 | 0,01 |

### Contrôles qualité

| Contrôle | Résultat |
|---|---|
| Valeurs manquantes, doublons | 0 |
| Heures sans météo | 0 |
| Part de week-end | 29 % (attendu 28,6 %) |
| `unmetered_wh` négatif | aucun |
| Profil de température en hiver | min 7 h, max 15 h |

---

## 7. Protocole d'évaluation

Commun à tous les modèles, pour que les résultats soient comparables :

| Élément | Choix |
|---|---|
| Découpage temporel | train 2007–2008, validation 2009, **test 2010** (toutes les saisons) |
| Normalisation | calculée sur le train uniquement |
| Entrées futures | calendrier seul (la température future observée n'est utilisée que dans une ablation « météo parfaite ») |
| Heures imputées | exclues de la perte et des métriques |
| Choix des hyperparamètres | sur la validation, avec 3 seeds |
| Test | utilisé **une seule fois**, à la fin |
| Baselines | persistance J-1, persistance J-7, profil moyen (jour × heure) |

---

## 8. Modèles et résultats

*À compléter au fur et à mesure.*

| Modèle | MAE (kW) | RMSE (kW) | WAPE (%) |
|---|---|---|---|
| Persistance J-1 | – | – | – |
| Persistance J-7 | – | – | – |
| Profil moyen | – | – | – |
| SARIMA | – | – | – |
| MLP | – | – | – |
| CNN | – | – | – |
| LSTM | – | – | – |
| TCN | – | – | – |

---

## 9. Avancement

- [x] Construction du dataset (nettoyage, agrégation, imputation, calendrier, météo)
- [ ] Analyse exploratoire (EDA)
- [ ] Baselines (persistance, profil moyen, SARIMA)
- [ ] MLP
- [ ] CNN
- [ ] LSTM
- [ ] Analyse des erreurs (horizon, pics, intervalles de confiance)

---

## 10. Limites connues

- **Une seule maison :** série très bruitée (four, lessive), ce qui limite la précision atteignable.
- **Heure fixe supposée (UTC+1)**, déduite des données et confirmée par le profil de température.
- **Météo :** réanalyse de la zone, pas un capteur sur place.
- **Calendrier :** jours fériés uniquement ; les vacances scolaires sont une piste d'amélioration.

---

## 11. Références

- G. Hébrail, A. Bérard – *Individual Household Electric Power Consumption*, UCI Machine Learning Repository. [doi:10.24432/C58K54](https://doi.org/10.24432/C58K54)
- [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api)
