# DoS Detection — Réseau de neurones from scratch

Détection d'attaques par déni de service (DoS) dans un réseau **SDN** (Software-Defined Network), à l'aide d'un réseau de neurones implémenté entièrement à la main — sans framework de deep learning.

## Présentation

Ce projet explore la détection d'anomalies réseau à partir de données de trafic SDN (`dataset_sdn.csv`). Après une phase d'exploration avec des approches classiques (Isolation Forest), le choix s'est porté sur un **réseau de neurones codé de zéro**, avec sa propre implémentation de la descente de gradient — sans `PyTorch` ni `TensorFlow`.

Le rapport détaillé de la démarche (choix des modèles, itérations, résultats) se trouve dans `rapport.pdf` à la racine du dépôt.

## Fonctionnalités

- **Prétraitement** des données de trafic réseau (nettoyage, normalisation)
- **Réseau de neurones from scratch** : forward pass, backpropagation et descente de gradient implémentés manuellement
- **Comparaison avec une approche classique** (Isolation Forest) pour évaluer la pertinence du deep learning sur ce problème
- **Visualisation** des résultats (courbes d'apprentissage, matrices de confusion) via `matplotlib`

## Prérequis

- Python 3
- Bibliothèques : `pandas`, `numpy`, `scikit-learn`, `matplotlib`

```bash
pip install pandas numpy scikit-learn matplotlib
```

## Installation et exécution

```bash
git clone https://github.com/antoinettefrmd/2026-dos_detection-neural_network.git
cd 2026-dos_detection-neural_network/src
python3 main.py
```

## Structure du projet

```
src/
├── main.py       # Point d'entrée : chargement des données, entraînement, évaluation
├── Proton.py      # Implémentation du réseau de neurones (forward/backward, descente de gradient)
└── dataset_sdn.csv  # Données de trafic réseau SDN utilisées pour l'entraînement
rapport.pdf        # Rapport détaillant la démarche, les choix de modèle et les résultats
```

## Démarche

Le développement a suivi une approche itérative :
1. Exploration des données et premières tentatives avec Isolation Forest
2. Constat des limites de l'approche classique sur ce jeu de données
3. Conception et implémentation d'un réseau de neurones simple
4. Ajout progressif de couches et ajustement des hyperparamètres

## Auteurs

Projet réalisé en binôme dans le cadre d'un cours de programmation / machine learning — Fonseca & Antoinette Fourmond.