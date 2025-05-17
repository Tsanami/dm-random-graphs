"""
Обёртки для вычисления статистик T^knn и T^dist с использованием GraphAnalyzer.
"""

import numpy as np
from src.build_graph import build_knn_graph, build_distance_graph
from src.graph_analyzer import GraphAnalyzer


def T_knn(data: np.ndarray, k: int) -> int:
    """
    T^knn = Δ(G) - максимальная степень в KNN-графе.
    """
    A = build_knn_graph(data, k)
    analyzer = GraphAnalyzer(A)
    return analyzer.max_degree()


def T_dist(data: np.ndarray, d: float) -> int:
    """
    T^dist = χ(G) - приближённое хроматическое число графа расстояний.
    """
    A = build_distance_graph(data, d)
    analyzer = GraphAnalyzer(A)
    return analyzer.chromatic_number()
