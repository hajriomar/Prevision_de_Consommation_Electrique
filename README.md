# Prévision de consommation électrique à 24 h

Prédire la consommation électrique **horaire** d'une maison sur les **24 prochaines heures**, à partir de l'historique, du calendrier et de la météo. Quatre modèles sont comparés sur un protocole commun : **MLP, CNN, LSTM et SARIMA**.

Projet Deep Learning – Polytech Annecy · Gharsallah Souha, Hajri Mohamed Omar

---

## Contenu du dépôt

| Fichier | Rôle |
|---|---|
| `Dataset_Prevision_de_Consommation_Electrique.ipynb` | Construit le dataset horaire et l'enregistre sur Google Drive |
| `MLP_Prevision_de_Consommation_Electrique.ipynb` | MLP : études d'hyperparamètres, modèle final, test |
| `CNN_Prevision_de_Consommation_Electrique.ipynb` | CNN 1D : études d'hyperparamètres, modèle final, test |
| `LSTM_Prevision_de_Consommation_Electrique.ipynb` | LSTM : réglage (8 configurations × 3 seeds), modèle final, test |
| `SARIMA_Prevision_de_Consommation_Electrique.ipynb` | SARIMA : 4 configurations, prévision chaque jour à minuit |

## Utilisation (Google Colab)

1. Lancer le **notebook Dataset** : il crée `dataset_hourly.parquet` dans `MyDrive/projet_conso/data/processed/`.
2. Lancer un **notebook de modèle**, avec le GPU activé pour MLP, CNN et LSTM. Il lit le dataset sur Drive, entraîne le modèle et l'évalue sur 2010.

Les données et les modèles ne sont pas dans le dépôt : ils sont générés par les notebooks.

---

## Données

- **Consommation :** [UCI – Individual Household Electric Power Consumption](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption), une maison à Sceaux, mesures à la minute de déc. 2006 à nov. 2010.
- **Météo :** température ressentie horaire à Sceaux, via [Open-Meteo](https://open-meteo.com/en/docs/historical-weather-api).
- **Jours fériés :** librairie `holidays` (France).

## Construction du dataset

1. **Nettoyage :** index `datetime`, valeurs `?` converties en manquantes.
2. **Agrégation horaire :** moyenne des minutes (somme pour les sous-compteurs) ; une heure est valide si au moins 45 minutes sont mesurées.
3. **Valeurs manquantes (436 h, 1,26 %) :** interpolation pour les trous de 3 h ou moins, sinon même heure la semaine précédente. Ces heures sont marquées `is_imputed` et exclues de la perte et des métriques.
4. **Variables ajoutées :** consommation non mesurée (`unmetered_wh`), heure, jour, mois, week-end, jour férié.
5. **Météo :** alignée en heure fixe UTC+1, comme les données UCI.

**Résultat :** `dataset_hourly.parquet`, **34 587 heures × 15 colonnes**, sans valeur manquante. Cible : `global_active_power` (kW).

---

## Protocole commun

- **Découpage dans le temps :** entraînement déc. 2006 → 2008, validation 2009, test 2010.
- **Réglage** sur la validation uniquement, avec early stopping ; le **test est utilisé une seule fois**.
- **Normalisation** calculée sur l'entraînement seulement.
- **Futur connu :** seul le calendrier des 24 h à prédire peut être utilisé. La météo future n'est pas connue à l'avance (testée en ablation sur le LSTM : aucun gain).
- **Métriques :** MAE et RMSE en kW, WAPE en % (erreur totale ÷ consommation totale).
- **Baselines :** persistance J-1 (veille), persistance J-7 (semaine précédente), profil moyen jour × heure.

## Modèles

| Modèle | Historique | Architecture | Perte | Réglages retenus |
|---|---|---|---|---|
| MLP | 48 h × 17 variables | aplatissement → 256 → 128 → 24 | MSE | dropout 0,3 · Adam, lr 1e-4 · batch 64 |
| CNN 1D | 168 h × 17 variables | 3 convolutions (32, 32, 64 filtres, noyau 3 h) + 2 max-pooling → dense 64 → 24 | MSE | dropout 0,3 · Adam, lr 1e-4 · batch 64 |
| LSTM | 336 h × 10 variables | LSTM 2 × 64 + calendrier des 24 h → dense 128 → 24 | MAE | dropout 0,4 · Adam, lr 1e-3 · batch 256 |
| SARIMA | toute la série passée | SARIMA(1,0,1)(0,1,1)₂₄, paramètres estimés sur 8 semaines puis figés | — | meilleure MAE et meilleur AIC sur 2009 |

Les trois réseaux prédisent les 24 heures d'un coup (prévision directe).

## Résultats (test 2010)

Une prévision de 24 h par jour, à minuit, sur 329 jours, avec les mêmes heures pour tous les modèles.

| Modèle | MAE (kW) | RMSE (kW) | WAPE (%) |
|---|---|---|---|
| Persistance J-1 | 0,558 | 0,823 | 52,6 |
| Persistance J-7 | 0,576 | 0,832 | 54,3 |
| Profil moyen | 0,513 | 0,687 | 48,3 |
| MLP | 0,467 | 0,626 | 44,0 |
| CNN | 0,453 | **0,623** | 42,7 |
| LSTM | **0,452** | 0,637 | **42,6** |
| SARIMA | 0,454 | 0,625 | 42,8 |

- **Tous les modèles battent les baselines.** LSTM, CNN et SARIMA sont au coude à coude, environ 12 % de mieux que le profil moyen. Le MLP suit.
- **Chaque modèle gagne sur la perte qu'il optimise :** perte MSE (MLP, CNN) → meilleure RMSE ; perte MAE (LSTM) → meilleure MAE.
- **Plus d'historique aide le CNN et le LSTM, pas le MLP**, qui perd l'ordre des heures en aplatissant la séquence.

> Les réseaux sont entraînés sur GPU : relancer un notebook peut faire varier les résultats de quelques millièmes de kW.

## Limites et pistes

- Une seule maison : série très bruitée (four, lessive), qui limite la précision atteignable.
- Écarts entre modèles proches de la variation d'un entraînement à l'autre.
- SARIMA estimé sur 8 semaines seulement, car le calcul est long.
- Pistes : plusieurs maisons, prévisions avec intervalles de confiance, analyse des erreurs par heure, modèle TCN.
