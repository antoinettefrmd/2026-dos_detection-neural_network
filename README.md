# Proposition de projet (révisée) - Algorithme autonome de détection d’anomalie basé sur des techniques de ML semi-supervisé et ensemble algorithmique -

## Problématique

Avec la constante croissance des échanges de données et la multiplication des systèmes connectés, la sécurité numérique est devenue un enjeu majeur au sein de notre société. En effet, des recherches montrent que 72% des compagnies ont subi des attaques de ransomware entre 2018 et 2023 dans le monde (Happe & Cito, [[2025](https://arxiv.org/html/2507.00829v1)]).    
Ce projet propose de concevoir un algorithme capable de détecter en temps réel une anomalie sur un réseau local, c'est-à-dire un accès inconnu, un pattern ou un data point dans le réseau avec une grande divergence du comportement historiquement défini (Knetik [[2025](https://www.kentik.com/kentipedia/network-anomaly-detection/)], VMware 2024).

## Objectif

Nous allons développer un algorithme intelligent, d'optimisation de poids, qui utilise la descente de gradient pour prétraiter les données d'un flux réseau. Ces données seront transmises à deux algorithmes de ML semi-supervisionnés Isolation Forest et Random Forest afin d'affiner leur détection des anomalies sur un réseau local. Une anomalie est définie par une grande déviation du comportement historiquement défini par la base de données. Elle est détectée par l'analyse de paquets sur un flux comportant les métriques suivantes : l'adresse IP, le numéro de port, le protocole, la taille du fragment, l'heure. Ces métriques peuvent changer, si nécessaire, au cours du projet. Ces métriques vont être pondérées pour une meilleure optimisation. On détecte ainsi un accès inconnu, un pattern ou un data point dans le réseau.

L'algorithme sera développé en définissant puis construisant un réseau de neurones simplifié (SNN). Au début, il y aura huit neurones, il est possible que ce nombre évolue au cours du projet. Nous seront ammené à calculer les dérivées partielles de chaque paramètres et effectuer une descente de gradient sur celles-ci. On implementra également l'optimisateur Adam.  technique d'isolation des données divergentes et qui aura comme biais un threshold défini par la baseline (l'historique de comportement normal du réseau). Il sera défini sur le SNN un algorithme d’optimisation par descente de gradient (Adam's method ?) et cooccurrence probabilistique (Markov chain) (Rimella, [[2025](https://arxiv.org/html/2004.06963v3https://arxiv.org/html/2004.06963v3)]). La descente de gradient sera responsable de minimiser les chances de logs négatifs (NLL), ce qui nous donnerait des poids plus optimaux ; ces poids seront utilisés dans un état de Markov caché (ht) qui prédira le résultat à partir de la fréquence d'apparition des data points.

L'algorithme va être entraîné à partir de plusieurs bases de données, qui seront responsables de définir son baseline normal (choisir quelle BDD utiliser pour ça), de définir le modèle d'une anomalie basé sur le flux des données (mettre la BDD sélectionnée), et de définir le modèle d'une anomalie basé sur le temps.

### Principal

1. Développement du réseau à 8 (ou plus) neurones ;
2. Implémentation d’analyse du trafic réseau par isolement des forêts ;
3. Optimisation des poids du SNN avec la méthode de descente par gradient Adam ;
4. Définition du threshold à partir d'une baseline.

### Intermédiaire

1. Optimisation d’analyse par prédiction probabiliste des poids/fréquences (Markov chain) ;

### Supplémentaire

1. Comparaison de la vitesse de convergence entre notre algorithme et un algorithme de machine learning supervisé tel que Random Forest à qui on donne des données labellisées directement sans prétraitement.
2. Comparaison de la vitesse de convergence entre notre algorithme et un algorithme de machine learning non supervisé.

## Testabilité

Essais sur des logiciels type … pour entraîner la robustesse de notre réseau et notre IA.

### Tests unitaires

- Pour chaque module du système (détection d’anomalie, module IA, mécanisme de réaction) ;
- Tests de validation fonctionnelle : vérification que les échanges entre modules sont corrects ;

### Tests de performance

- Taux de détection : proportion d’attaques bien identifiées ;
- Taux de faux positifs : proportion d’alertes déclenchées à tort ;
- Temps de réponse : délai entre la détection et la réponse automatique.

## Test de sécurité

Validation de la capacité du module de défense.