# ------------------------------------------------------------------
# Ce script réalise des tests d’ANOVA à un facteur (X3 ou X6) et à deux facteurs (X3 et X6)
# sur la variable Y1 du dataset ENB2012_data.xlsx. L’objectif est de vérifier si les variables
# catégorielles X3 et/ou X6 influencent significativement la moyenne de Y1.
# Un test ANOVA (analyse de la variance) est utilisé pour comparer les moyennes
# de plusieurs groupes et tester l’hypothèse H0 : "toutes les moyennes sont égales".
# ------------------------------------------------------------------

import pandas as pd                    # Librairie pour la manipulation de données
import scipy.stats as stats           # Pour effectuer le test ANOVA
from google.colab import files        # Pour uploader le fichier depuis Google Colab
uploaded = files.upload()             # Lancement du téléchargement du fichier Excel

# Chargement du fichier Excel contenant les données
df = pd.read_excel("ENB2012_data.xlsx")

# --- ANOVA à un facteur : influence de X3 sur Y1 ---

# On regroupe les valeurs de Y1 selon chaque modalité (valeur unique) de X3
groupes = [df[df["X3"] == valeur]["Y1"] for valeur in df["X3"].unique()]

# Test ANOVA à un facteur : H0 = toutes les moyennes sont égales entre les groupes
stat, p_value = stats.f_oneway(*groupes)

# Affichage de la statistique F et de la p-value
print(f"Statistique du test : {stat:.3f}")
print(f"Valeur p : {p_value:.3f}")

# Interprétation du test avec un seuil alpha = 0.05
if p_value < 0.05:
    print("Rejet de H0 : Le facteur X3 influence significativement Y1.")
else:
    print("Acceptation de H0 : Pas de différence significative des moyennes.")

# --- ANOVA à un facteur : influence de X6 sur Y1 ---

import pandas as pd
import scipy.stats as stats
from google.colab import files
uploaded = files.upload()

df = pd.read_excel("ENB2012_data.xlsx")

# On regroupe les valeurs de Y1 selon chaque modalité de X6
groupes = [df[df["X6"] == valeur]["Y1"] for valeur in df["X6"].unique()]

# Test ANOVA : H0 = pas de différence entre les moyennes selon X6
stat, p_value = stats.f_oneway(*groupes)

print(f"Statistique du test : {stat:.3f}")
print(f"Valeur p : {p_value:.3f}")

if p_value < 0.05:
    print("Rejet de H0 : Le facteur X6 influence significativement Y1.")
else:
    print("Acceptation de H0 : Pas de différence significative des moyennes.")

# --- ANOVA à deux facteurs : interaction entre X3 et X6 sur Y1 ---

import pandas as pd
import statsmodels.api as sm                # Pour les modèles statistiques
import statsmodels.formula.api as smf       # Pour écrire des formules de modèles
from google.colab import files
uploaded = files.upload()

df = pd.read_excel("ENB2012_data.xlsx")

# On crée un modèle linéaire avec interaction entre X3 et X6
# C(X3) et C(X6) signifient que ces variables sont traitées comme catégorielles
model = smf.ols("Y1 ~ C(X3) * C(X6)", data=df).fit()

# On effectue l’analyse de la variance (ANOVA) de type II
anova_table = sm.stats.anova_lm(model, typ=2)

# Affichage du tableau ANOVA qui donne :
# - les sources de variation (C(X3), C(X6), interaction C(X3):C(X6))
# - la somme des carrés (sum_sq)
# - les degrés de liberté (df)
# - la statistique F
# - la p-value (PR(>F))
print(anova_table)
