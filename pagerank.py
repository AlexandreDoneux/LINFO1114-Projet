# fichiers contenants les différentes méthodes de calcul du PageRank

import numpy as np

def pageRankLinear(A: np.ndarray, alpha: float, v: np.ndarray) -> np.ndarray:
    """
    Calcule le PageRank via résolution d'équations linéaire

    Args:
        A (np.ndarray): Matrice d'adjacence.
        alpha (float): parametre de téléportation (compris entre 0 et 1).
        v (np.ndarray): Vecteur de personnalisation.

    Returns:
        np.ndarray: scores pagerank (dans le même ordre que les lignes de la matrice d’adjacence)
    """
    pass


def pageRankPower(A: np.ndarray, alpha: float, v: np.ndarray) -> np.ndarray:
    """
    Calcule le PageRank en utilisant lla Power method.

    Args:
        A (np.ndarray): Matrice d'adjacence.
        alpha (float): parametre de téléportation (compris entre 0 et 1).
        v (np.ndarray): Vecteur de personnalisation.

    Returns:
        np.ndarray: scores pagerank (dans le même ordre que les lignes de la matrice d’adjacence)
    """
    
    n = A.shape[0]
    epsilon = 1e-8
    max_iter = 10000
    
    
    
    v = v.reshape((n, 1))
    # Normalisation de v 
    if np.sum(v) != 0:
        v = v / np.sum(v)
    else:
        v = np.ones((n, 1)) / n

    P = np.array(A, dtype=float)
    
    for i in range(n):
        row_sum = np.sum(P[i, :])
        if row_sum == 0:
            # Gestion Noeuds sans issue
            P[i, :] = 1.0 / n
        else:
            
            P[i, :] /= row_sum

    # La formule est G = alpha * P + (1 - alpha) * e * v^T
    e = np.ones((n, 1))
    
    G = alpha * P + (1 - alpha) * np.dot(e, v.T)

    print("\n Matrice d'adjacence A :\n", A)
    print("\n Matrice de probabilite de transition P :\n", P)
    print("\n Matrice Google G :\n", G)

    x = np.sum(A, axis=0).reshape((n, 1)).astype(float)
    
    if np.sum(x) == 0:
        x = np.ones((n, 1)) / n
    else:
        x = x / np.sum(x)

    print("\n Les trois premieres iterations de la power method :")

    for k in range(max_iter):
        x_old = x.copy()

        # Lormule matricielle : x(k+1) = G^T * x(k)
        x = np.dot(G.T, x_old)

        x = x / np.sum(x)

        if k < 3:
            print(f"  Iteration {k+1} : {x.flatten()}")

        error = np.sum(np.abs(x - x_old))
        
        if error < epsilon:
            break
            
    print(f"\nConvergence atteinte apres {k+1} iterations.")
    print(" Resultat final  :", x.flatten())
    
    return x.flatten()
   


def randomWalk(A: np.ndarray, alpha: float, v: np.ndarray) -> np.ndarray:
    """
    Calcule le PageRank en utilisant un marcheur aléatoire.

    Args:
        A (np.ndarray): Matrice d'adjacence.
        alpha (float): parametre de téléportation (compris entre 0 et 1).
        v (np.ndarray): Vecteur de personnalisation.

    Returns:
        np.ndarray: scores pagerank (dans le même ordre que les lignes de la matrice d’adjacence)
    """
    pass