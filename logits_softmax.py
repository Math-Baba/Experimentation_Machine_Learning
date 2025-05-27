# Importation des bibliothèques nécessaires
import pandas as pd                     # Pour la manipulation des données (DataFrame)
import numpy as np                      # Pour le calcul numérique (matrices, vecteurs)
import matplotlib.pyplot as plt         # Pour l'affichage graphique
import seaborn as sns                   # Pour des graphiques plus esthétiques

# Chargement du fichier Excel depuis l'interface utilisateur (utile sur Google Colab)
uploaded = files.upload()

# 1. Lecture du fichier Excel et création d’un DataFrame
df = pd.read_excel("ENB2012_data.xlsx")
df.head()  # Affiche les 5 premières lignes pour aperçu

# 2. Création d’une nouvelle colonne "Classe"
# Si la consommation de chauffage (Y1) est supérieure à celle de refroidissement (Y2), alors Classe = 0, sinon 1
df['Classe'] = np.where(df['Y1'] > df['Y2'], 0, 1)

# 3. Séparation des données d’entrée (features) et des étiquettes (labels)
# Les variables explicatives (X1 à X8) sont les caractéristiques des bâtiments
X = df[['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8']].values
y = df['Classe'].values   # Les classes cibles (0 ou 1)
n_classes = 2             # Nombre de classes : chauffage dominant ou refroidissement dominant
n_features = X.shape[1]   # Nombre de caractéristiques (colonnes X1 à X8)

# 4. Initialisation aléatoire des poids (W) et biais (b) pour chaque classe
np.random.seed(0)                       # Fixe la graine pour avoir les mêmes résultats à chaque exécution
W = np.random.randn(n_classes, n_features)  # Poids aléatoires pour chaque classe et chaque feature
b = np.random.randn(n_classes)              # Biais aléatoires pour chaque classe

# 5. Calcul des logits (sortie brute du modèle linéaire avant softmax)
logits = np.dot(X, W.T) + b
print("Logits (extraits) :\n", logits[:5])  # Affiche les 5 premières lignes des logits

# 6. Définition de la fonction softmax
# Convertit les logits en probabilités normalisées pour chaque classe
def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # Pour la stabilité numérique
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

# 7. Calcul des probabilités avec softmax
probs = softmax(logits)
print("\nProbabilités (softmax) :\n", probs[:5])  # Affiche les 5 premières lignes des probabilités

# 8. Prédiction : on choisit la classe avec la probabilité la plus élevée
y_pred = np.argmax(probs, axis=1)

# 9. Évaluation du modèle : pourcentage de bonnes prédictions (exactitude)
accuracy = np.mean(y_pred == y)
print(f"\nExactitude du modèle : {accuracy * 100:.2f}%")

# 10. Affichage de l’histogramme des classes
plt.figure(figsize=(6, 4))
sns.countplot(x=df['Classe'])  # Affiche le nombre d’échantillons pour chaque classe
plt.title("Répartition des classes : chauffage vs refroidissement")
plt.xlabel("Classe (0 = Chauffage dominant, 1 = Refroidissement dominant)")
plt.ylabel("Nombre d'échantillons")
plt.tight_layout()
plt.show()
