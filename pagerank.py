# fichiers contenants les différentes méthodes de calcul du PageRank

import numpy as np

def prepare_data(A: np.ndarray, alpha: float, v: np.ndarray) -> np.ndarray:
    """
    Prépare les données pour le calcul du PageRank en construisant la matrice Google G.
    Args:
        A (np.ndarray): Matrice d'adjacence.
        alpha (float): parametre de téléportation (compris entre 0 et 1).
        v (np.ndarray): Vecteur de personnalisation.
    Returns:
        np.ndarray: Matrice Google G.
    """
    n = A.shape[0]

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
            P[i, :] = 1.0 / n  # On distribue uniformément la probabilité ? On ne veut pas une ligne de 0 ? -> vérifier
        else:

            P[i, :] /= row_sum

    # La formule est G = alpha * P + (1 - alpha) * e * v^T
    e = np.ones((n, 1))

    G = alpha * P + (1 - alpha) * np.dot(e, v.T)  # v.T ?
    return G,P 

# --------------------------------------------------------------------------------------------------------------------

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
    n = A.shape[0]
    G, P = prepare_data(A, alpha, v)
    I = np.eye(n)

    # (I - alpha P^T) x = (1 - alpha) v
    # Extract P from G (see slide
    P = (G - (1 - alpha) * np.ones((n, 1)) @ v.reshape(1, n)) / alpha

    # Normalisation de v
    v = v.reshape((n, 1))
    if np.sum(v) != 0:
        v = v / np.sum(v)
    else:
        v = np.ones((n, 1)) / n

    # Linear system + solve
    if alpha < 1:
        # (I - alpha.P^T).x = (1 - alpha).v
        M = np.eye(n) - alpha * P.T
        b = (1 - alpha) * v
    else:
        # (I - P^T) x = 0 avec sum(x) = 1
        M = np.eye(n) - P.T
        b = np.zeros((n, 1))

        # Replace last equation with sum(x) = 1
        M[-1, :] = np.ones(n)
        b[-1, 0] = 1.0

    x = np.linalg.solve(M, b)

    # Normalisation pour éviter les problèmes de calculs
    x = x / np.sum(x)

    return x.flatten()


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
    
    
    G, P = prepare_data(A, alpha, v)

    print("\n Matrice d'adjacence A :\n", A)
    print("\n Matrice de probabilite de transition P :\n", P)
    print("\n Matrice Google G :\n", G)

    x = np.sum(A, axis=0).reshape((n, 1)).astype(float) # ?
    
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
    n = A.shape[0]
    n_steps = 10001
    burn_in = 1000

    # PageRank exact
    p_star = pageRankLinear(A, alpha, v)

    # Matrices
    G, P = prepare_data(A, alpha, v)

    # Normalisation de v
    v = v / np.sum(v)

    visits = np.zeros(n)
    errors = np.zeros(n_steps)

    current = 0  # noeud A

    for k in range(n_steps):
        if np.random.rand() < alpha:
            current = np.random.choice(n, p=P[current, :])
        else:
            current = np.random.choice(n, p=v)

        if k >= burn_in:
            visits[current] += 1
            p_rw = visits / np.sum(visits)
            errors[k] = np.mean(np.abs(p_rw - p_star))

    print("\nRésultat final (random walk) :")
    print(p_rw)

    print("\nDonnées (k, ε(k)) :")
    for k in range(burn_in, n_steps, (n_steps - burn_in) // 20):
        print(f"{k} {errors[k]}")

    return p_rw