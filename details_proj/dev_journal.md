# Journal personel - status du projet

## Prototype (protoN.py)

Dans cette section on vais discuter le status du developpement du prototype d'une simple IA.

### Neuron

La classe "Neuron" définit un perceptron que prendre un n valeurs comment entrée, et sort un valeur d'évaluation (score)
à parti d'un seuil (threshold) dynamique. Cette classe est partgé dans differents méthodes:
    - sigmoid : function d'activation borné entre [ 0-1 ]. Prendre comme parametre la somme des datas d'entrées avec ses
    poids et bias et retourne un valeur entre 0 et 1.
        Formula : $\σ(x)=1/(1+exp(-x))

### Problems

Un prémier problème que le prototype à reporté, c'était en relation à la précision de decision (loss function). C'était remarque que,
dans le prémiers tests, la précision de perte descendait drasticament de 34% à 0% de la prémier interation pour la deuxième.
Le binôme a arrivé sur deux possible hypotèses d'origine de cette error:
    1. mauvais application de la fonction d'évaluation d'entropie (log_loss);
    2. partition incorrect ou problème sur l'envoie des data.
