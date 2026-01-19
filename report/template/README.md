# Information sur le rapport intermediaire

Ce fichier contient les instructions pour rédiger
le rapport d'avancement de votre projet.


## Format

Le rapport doit être redigé en utilisant LaTeX ([https://en.wikipedia.org/wiki/LaTeX](https://en.wikipedia.org/wiki/LaTeX)).

Le fichier principal doit s'appeller `rapport-nom1-nom2.tex`,
où `nom1` et `nom2` sont les noms des membres du binôme.
Ce fichier doit se trouver dans le répértoire 

`report`

qui se trouve dans votre dépôt git.

Le fichier `rapport-nom1-nom2.tex` doit être compilable
par la commande 

```
pdflatex rapport-nom1-nom2.tex
```

si ce n'est pas le cas, alors vous devez fournir aussi
un `Makefile` pour comlpiler votre rapport par la commande

```
make
```

Le fichier LaTeX doit avoir comme classe _article_, i.e.
commencer par

```
\documentclass{article}
```

Le pdf qui resulte de la compilation doit être __au maximum de 4 pages__.
En cas de doute vous pouvez trouver cette information à l'aide 
de la commande `pdfinfo`.


## LaTeX

### Dépendances

Vous devez installer dans votre OS la commande `pdflatex`,
et toute les dépendances necessaires.

Sur Ubuntu il suffit de éxécuter la commande

```
apt-get install texlive-latex-base
```

possiblement avec `sudo`.

LaTeX est un language très riche, qui dispose de  dizaines de packages.
Nous vous suggérons d'installer tous les packages requis pour compiler
le document d'exemple que vous pouvez trouver dans le répertoire `report/template/`.

### Example

Le répertoire `report/template/` contient un fichier LaTeX
et un `Makefile` pour le compiler.
L'exécution de la commande `make` dans le répertoire `report/template/`,
_devrait_ générer le fichier `rapport.pdf`.

Si l'exécution de `make` échoue, lisez les messages d'erreur.

Si l'échec est dû à un package LaTeX manquant,
installez-le et éxecutez à nouveau `make`.

Si l'échec est dû à toute autre raison,
vous pouvez nous contacter par mail pour
démander de l'aide.



## Contenu du rapport

Le rapport doit être divisé en trois parties

- [ ] Introduction :

  - Quel est le but de votre projet ? 
  - Métrique :
  Comment mesurer si le projet est un succès ou un échec ?
  
  Répondre à cette question est très important :
  si on ne dispose pas de métrique pour juger
  du succès d'un projet, il est considéré comme échoué, tout court.

- [ ] Implémentation :

  - Comment êtes-vous en train de réaliser votre projet ?
  - Quel logiciel avez-vous déjà codé ?
  - Quelle est sa structure en modules / packages ?
  - Quelle est la représentation des données ?
  - Quelles sont les technologies sur lesquelles votre code s'appuie ?
  
- [ ] Jalons :
  - Quelles sont les tâches à réaliser pour mener le projet à sa conclusion ?
  - Quelle est l'organisation temporelle des tâches ?
  - Quelles tâches sont déjà terminées ?
  
  Le schéma dans le transparent 7 du [pdf de présentation du cours](https://www.irif.fr/~gio/teaching/2025-26/pprog/presentation-2526.pdf)
  montre un schéma d'organisation temporelle.



## Date du rendu

La version finale du rapport doit être dans votre dépôt git
au plus tard pour le

__28 Fév à 23h59__


## Note

Nous allons noter les points suivants:

1. présentation du projet
1. de ce qui a été fait
1. des difficultés rencontrées
1. de ce qui va être fait
1. forme du rapport

C'est vraiment le contenu du rapport qui va être noté, indépendamment
de la qualité ou de l'avancement du projet
(à condition que le rapport décrive en effet la réalité).

