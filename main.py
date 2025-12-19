# Fichier de lecture des CSV et lançant les algorithmes Pagerank différents :
# - résolution équations linéaires
# - Power method
# - Marcheur aléatoire

import numpy as np
from pprint import pprint

from pagerank import pageRankLinear, pageRankPower, randomWalk

# read adjacency matrix, personalization vector from CSV files
A = np.loadtxt('matrice_adjacente.csv', delimiter=',', dtype=int)
v = np.loadtxt('VecteurPersonnalisation_Groupe25.csv', delimiter=',', dtype=float)

#pprint(A)
#pprint(v)

# quel paramètre de téléportation choisir ?
alpha = 0.9
#alpha = 1
# problème lorsque alpha = 1 pour résolution d'équation linéaire. On doit modifier la dernière équation pour imposer la somme des scores égale à 1. (e^T.x = 1)

print("Calcul du PageRank par résolution d'équations linéaires : ---------------------------------------------")
pr_linear = pageRankLinear(A, alpha, v)
pprint(pr_linear)

print("\nCalcul du PageRank par Power Method : ---------------------------------------------")
pr_power = pageRankPower(A, alpha, v)
pprint(pr_power)


print("\nCalcul du PageRank par Marcheur Aléatoire : ---------------------------------------------")
pr_random_walk = randomWalk(A, alpha, v)
pprint(pr_random_walk)



print("\nComparaison des résultats : ---------------------------------------------")
print("Différences entre PageRank Linéaire et Power Method :")
diff_linear_power = np.abs(pr_linear - pr_power)
pprint(diff_linear_power)

print("\nDifférences entre PageRank Linéaire et Marcheur Aléatoire (final) :")
diff_linear_random = np.abs(pr_linear - pr_random_walk)
pprint(diff_linear_random)