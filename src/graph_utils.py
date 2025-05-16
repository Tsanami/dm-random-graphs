"""
graph_utils.py
Модуль для построения графов и вычисления статистик: KNN, пороговых, максимальная степень и жадное хроматическое число.
"""
import numpy as np
import networkx as nx
from sklearn.neighbors import NearestNeighbors


def build_knn_graph(X: np.ndarray, k: int) -> np.ndarray:
    """
    Строит KNN-граф на данных X и возвращает его матрицу смежности.

    Параметры:
    ----------
    X : np.ndarray, форма (n_samples, n_features)
        Входные данные.
    k : int
        Число соседей.

    Возвращает:
    -------
    adjacency : np.ndarray, shape (n_samples, n_samples)
        Матрица смежности (0/1).
    """
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(X.reshape(-1,1))
    A = nbrs.kneighbors_graph(X.reshape(-1,1)).toarray().astype(int)
    np.fill_diagonal(A, 0)
    return A


def build_distance_graph(X: np.ndarray, threshold: float) -> np.ndarray:
    """
    Строит граф пороговых расстояний: ребро добавляется,
    если евклидово расстояние <= threshold.

    Параметры:
    ----------
    X : np.ndarray, форма (n_samples, n_features)
        Входные данные.
    threshold : float
        Пороговое значение для ребра.

    Возвращает:
    -------
    adjacency : np.ndarray, shape (n_samples, n_samples)
        Матрица смежности (0/1).
    """
    dist_matrix = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
    A = ((dist_matrix <= threshold) & (dist_matrix > 0)).astype(int)
    return A


def adjacency_to_graph(A: np.ndarray) -> nx.Graph:
    """
    Преобразует матрицу смежности в networkx.Graph.
    """
    G = nx.from_numpy_array(A)
    return G


def max_degree(A: np.ndarray) -> int:
    """
    Вычисляет максимальную степень узла графа по матрице смежности.
    """
    degrees = A.sum(axis=1)
    return int(degrees.max())


def greedy_chromatic_number(A: np.ndarray) -> int:
    """
    Возвращает приближённое хроматическое число, полученное
    жадным алгоритмом раскраски.
    """
    G = adjacency_to_graph(A)
    coloring = nx.coloring.greedy_color(G, strategy='largest_first')
    return max(coloring.values()) + 1
