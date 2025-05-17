import networkx as nx
from sklearn.neighbors import NearestNeighbors
import numpy as np
import networkx as nx


def build_knn_graph(data: np.ndarray, k: int) -> nx.Graph:
    """
    Строит KNN‑граф в.
    Каждая точка соединяется с k ближайшими (без само‑петель).
    Узлы пронумерованы от 0 до len(data)-1.
    """
    if k <= 0:
        raise ValueError("k должно быть положительным.")

    G = nx.Graph()

    for i, coord in enumerate(data):
        G.add_node(i, x=float(coord))

    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(data.reshape(-1, 1))
    # distances[i], indices[i] — k+1 ближайших (включая саму точку)
    _, indices = nbrs.kneighbors(data.reshape(-1, 1))

    for i, neighs in enumerate(indices):
        for j in neighs:
            if i != j:
                G.add_edge(i, j)
    return G


def build_distance_graph(data: np.ndarray, d: float) -> nx.Graph:
    """
    Строит граф по расстоянию d.
    Проводит ребро между i и j, если |data[i] - data[j]| <= d.
    """
    if d <= 0:
        raise ValueError("Параметр d должен быть положительным.")

    n = data.shape[0]
    G = nx.Graph()

    for i, coord in enumerate(data):
        G.add_node(i, x=float(coord))

    # проверяем все пары (i, j) с i < j
    for i in range(n):
        for j in range(i + 1, n):
            if abs(data[i] - data[j]) <= d:
                G.add_edge(i, j)

    if G.number_of_edges() == 0:
        print("[WARNING] Все вершины изолированы при данном d.")

    return G
