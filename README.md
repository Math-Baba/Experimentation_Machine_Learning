# Projet Machine Learning - Expérimentations en Python

## Introduction
Ce projet regroupe plusieurs expérimentations en **machine learning**, réalisées sous **Google Colab**, dans un **objectif d’apprentissage**. Chaque implémentation explore une **approche statistique ou probabiliste**, mais aucune n'est optimisée à ce stade. L'idée est de tester différentes méthodes et analyser les résultats obtenus.

## Contenu du Projet
### 1. **Régression Logistique Multiclasse**
Ce modèle vise à prédire si un bâtiment a une charge de **chauffage** ou **refroidissement** dominante en fonction de plusieurs paramètres. L'algorithme suit les étapes suivantes :
- **Lecture et préparation des données** (extraction des caractéristiques X1 à X8).
- **Classification binaire** : si Y1 > Y2 alors Classe = 0 (chauffage), sinon Classe = 1 (refroidissement).
- **Calcul des logits et application de la fonction softmax** pour convertir les sorties en probabilités.
- **Prédiction des classes et calcul de l'exactitude** du modèle.
- **Visualisation** de la répartition des classes.

Lien vers le dataset : https://archive.ics.uci.edu/dataset/242/energy+efficiency
📄 **Code source** : `logits_softmax.py`

📝 **Observations** :
- Le modèle attribue **presque toute la probabilité à la classe chauffage**, indiquant qu’il doit être entraîné pour améliorer ses prédictions.
- **L’ajout de régularisation** et **d’un entraînement sur plusieurs itérations** pourraient améliorer les performances.

---

### 2. **Analyse Probabiliste d’un Corpus Littéraire**
Ce script utilise **spaCy** et **NLTK** pour explorer un corpus textuel et en extraire des probabilités de mots et de phrases :
- **Nettoyage du texte** (suppression des stopwords, ponctuation, mise en minuscules).
- **Génération d'un nuage de mots** pour identifier les termes les plus fréquents.
- **Calcul des probabilités unigramme, bigramme et trigramme**.
- **Évaluation de la probabilité d’une phrase** donnée en utilisant un modèle trigramme.

Fichier utilisé : PG.txt
📄 **Code source** : `proba_corpus.py`

📝 **Observations** :
- Le modèle permet d’analyser la structure probabiliste du langage.
- **Utiliser une plus grande quantité de données** pourrait aussi améliorer la précision des probabilités.

---

### 3. **Classification avec le Classificateur Naïf Bayes**
Ce projet compare une **implémentation manuelle** du classificateur **Naïf Bayes** avec la version de **Scikit-Learn (GaussianNB)** :
- **Calcul des probabilités conditionnelles** pour chaque classe selon une loi normale.
- **Discrétisation des données** (transformation de la variable continue Y1 en classes).
- **Entraînement et prédiction des classes** sur un dataset de consommation énergétique.
- **Comparaison des résultats** entre la version manuelle et celle optimisée avec Scikit-Learn.

📄 **Code source** : `prob_bayes.py`

📝 **Observations** :
- L’implémentation manuelle fonctionne mais **est beaucoup plus lente et sujette aux erreurs**.
- La version Scikit-Learn est plus rapide.

---

### 4. **Tests ANOVA (Analyse de Variance)**
Ce projet explore les **tests ANOVA à un facteur et à deux facteurs**, pour analyser l'effet des variables **X3 (type d’engrais)** et **X6** sur **Y1 (rendement)** :
- **ANOVA à un facteur** : teste si X3 ou X6 influencent significativement Y1.
- **ANOVA à deux facteurs** : analyse l’effet combiné de X3 et X6, et leur éventuelle interaction.

📄 **Code source** : `anova.py`

📝 **Observations** :
- **X3 influence significativement Y1**, mais **X6 et l'interaction X3:X6 n’ont pas d’effet mesurable**.
- **Respecter les conditions d’application de l’ANOVA** (normalité, homogénéité des variances) est crucial pour éviter des conclusions erronées.

---

## Conclusion
Ce projet m'a permis d’explorer plusieurs **approches en machine learning et en statistiques**, allant du **traitement probabiliste du langage** à la **modélisation statistique**. Il constitue une **base d’apprentissage**, avec plusieurs pistes d'amélioration possibles.

## Auteur
- **Math-Baba** - [GitHub](https://github.com/Math-Baba)
