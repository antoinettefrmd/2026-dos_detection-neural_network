# Algorithmes d’apprentissage

La détection d’anomalie contient deux branches importantes :

# Outlier Detection

Détecte les anomalies dans le train_set (Isolation Forest)

# Novelty detection

une novelty detection consiste à apprendre à partir de données normales uniquement, puis à détecter si de nouvelles observations diffèrent significativement de ce qui a été observé pendant l’entraînement.
Novelty detection → données d’apprentissage normales
Anomaly detection → données d’apprentissage mixtes (normales + anomalies)

# Apprentissage sans supervision

## Isolation Forest

l’Isolation Forest, un algorithme de Machine Learning destiné à résoudre des problèmes de classification binaires tels que la détection d’anomalies basée sur l’isolation. (date :Eighth IEEE International Conference en 2008)

détecter dans notre dataset les échantillons dont les caractéristiques x sont très éloignées de celles des autres échantillons. 

On peut calculer la moyenne et l’écar-type d’un échantillon pour calculer une fonction **densité de probabilité (loi Normale)** qui nous sert ensuite à calculer la probabilité qu’une donnée appartienne à un échantillon. 

Principe : On places les données sur un champs (repère orthogonal) en fonction d’une caractérisque. On effectue une série de splits aléatoires, partitions du champs en deux, en vérifiant à chaque fois si on a réussi à isoler une donnée. 

Moins on partitionne, cad moins on split, plus il est probable que la donnée isolée trouvée soit effectivement une anomalie. Et réciproquement, plus on doit spliter les échantillons pour trouver une donnée, plus celle-ci est conforme et donc il est moins probable que ce soit une anomalie.

Attention ! Il peut arriver qu’un des premiers splits isole une donnée qui n’est pas une anomalie. C’est très peu probable. Pour être sur que ce soit le cas ⇒ Technique d’ensemble : générer plusieurs estimateur avec le même algorithme et comparer l’ensemble pour être sûr de ne pas écarter une bonne donnée.

La construction de l’algorithme donne lieu à une structure d’arbre, dont les nœuds sont les ensembles partitionnés durant les étapes de l’algorithme et les feuilles sont les points isolés. Intuitivement les anomalies auront tendance à être les feuilles les plus proches de la racine de l’arbre.

La forêt est donc un ensemble d’arbres, donc un ensemble de données évaluées sur plusieurs caractérisques, qui permettent de trouver une anomalie sous plusieurs angles.

POINTS FORT : 

- capable d’analyser des datasets de très grandes dimentions, dataset ayant beaucoup de variables.
- robuste et précis, le nombre d’arbres peut être ajusté
- performant : la taille du sous-échantillonnage, ou le nombre de points de données utilisés pour construire chaque arbre, peut également varier pour améliorer les performances du modèle.
- facile à mettre en œuvre et à régler

POINTS FAIBLES :

- Sensibilité au nombre d’arbres
    - Trop peu d’arbres → modèle instable, résultats très variables.
    - Trop d’arbres → temps de calcul élevé, sans amélioration significative des performances.
- Sensibilité à la taille du sous-échantillonnage
    
    Chaque arbre est entraîné sur un sous-échantillon aléatoire du jeu de données.
    
    - Si le sous-échantillon est trop petit, les arbres ne captent pas bien la structure globale → trop de faux positifs (points normaux considérés comme anomalies).
    - Si le sous-échantillon est trop grand, les anomalies peuvent être “noyées” parmi les points normaux → faux négatifs (vraies anomalies non détectées).

IMPLEMENTATION

Python : 

- bibliothèques telles que Scikit-learn
- classe IsolationForest

src : [Machine Learnia](https://youtu.be/FTtzd31IAOw?si=2t6hpiyWibBoi95u)

[https://datascientest.com/isolation-forest](https://datascientest.com/isolation-forest)

[https://fr.statisticseasily.com/glossaire/qu'est-ce-que-la-forêt-d'isolement/](https://fr.statisticseasily.com/glossaire/qu%27est-ce-que-la-for%C3%AAt-d%27isolement/)

anomalie (communément appelée outliers)

## Local outlier Factor (LOF)

Novelty detection : le mécanisme par lequel un organisme intelligent est capable d'identifier un modèle sensoriel entrant comme étant jusqu'ici inconnu.

est un algorithme proposé par Markus M. Breunig, Hans-Peter Kriegel, Raymond T. Ng et Jörg Sander en 2000 pour la recherche de points de données anormaux en mesurant l'écart local d'un point de données donné par rapport à ses voisins.

### Principe :

Compare la densité locale d’un point à celle de ses voisins

src : [https://youtu.be/Ymvq6JHjoBY?si=V_r70v-15jHunI85](https://youtu.be/Ymvq6JHjoBY?si=V_r70v-15jHunI85)

# Apprentissage sous supervision

## Support Vector Machine (SVM)

## Random Forest (RF)

# Apprentissage sous semi-supervision

## Principal Component Analysis (PCA)

![Capture d’écran du 2025-11-12 22-35-21.png](Algorithmes%20d%E2%80%99apprentissage/Capture_dcran_du_2025-11-12_22-35-21.png)
