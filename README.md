# Proposition de projet - Algorithme intelligent de détection et de réaction automatique face aux cyberattaques dans un réseau local -

## Problématique

Avec la constante croissance des échanges de données et la multiplication des systèmes connectés, la sécurité numérique est devenu un enjeu majeur au sein de notre société. En effet, surveys montrent une estimative de que 72% des companies ont subid des attaques de ransomware entre 2018 et 2023 (Happe & Cito, [[2025](https://arxiv.org/html/2507.00829v1)]).
Un des défis présents dans le champ de la sécurité informatique est la forte nécessité de la présence humaine dans l'analyse et la détection des anomalies sur le réseau, même avec la présence des modèles de LLM et techniques en LM.
Ce projet propose de concevoir un analisateur des accèes autonome capable de détecter en 
temps réel une anomalie sur un reseau local, ça veut dire un accèes unconnu un pattern ou un data point dedans le reseau avec une grande descrepancia de le comportement historiquement défini (Knetik [[2025](https://www.kentik.com/kentipedia/network-anomaly-detection/)], VMware 2024), et de déclencher 
une réponse defensive.

## Objective

Le but du projet est de développer un système autonome en utilisant des méthodes de ML (machine learning) aussi qu'une LLM model, capable de : 

### Principal

1. Surveiller et analyser le traffic réseau en temps réel;
2. Détecter automatiquement les comportements suspects et malveillants, tels que les détections d'IPs non enregistrées, les scans de ports ou des attaques DDOS;
3. Protéger les données sensibles (eg. mécanisme de verrouillage, de chiffrement ou d'isolation automatique);

### Intérmediare
1. Déclencher une contre-mesure intelligente (eg. filtrage dynamique, blocage d'addresse, réponse simulée)
2. Implémentation d'un 

### Supplémentaire

1. Extension du modèle à un réseaux distribué (détection collaborative);
2. Implémentation d'un modèle de contre-intelligence offensive vers l'attaquant (eg. HoneyNets, Beaconing implants)

## Testabilité 

essais sur des logiciel type .. pour entrainé la robustesse de notre réseau et notre ia 

### Tests Unitaire
 
- Pour chaque module du système (detection d'anomalie, module l'IA, mécanisme de réaction);
- Tests de validation fonctionelle: verification de que les échanges entre modules sont correctes;

### Tests de performance

- Taux de détection : proportion d'attaques bien identifiées;
- Taux de faux positif : proportion d'alertes déclacher à tort;
- Temps de réponse : délai entre la détection et la réponse automatique.

# Test de sécurité

Validation de la capacité du module de défense
Test de vérification qu'aucune contre-mesure ne provoque d'effet secondaire indésirable sur le trafic.
