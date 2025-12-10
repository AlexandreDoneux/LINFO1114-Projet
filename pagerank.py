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
    pass


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