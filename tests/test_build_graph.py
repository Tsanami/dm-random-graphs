import numpy as np
from src.build_graph import build_knn_graph, build_distance_graph


def test_build_knn_graph_small():
    data = np.array([0.0, 1.0, 2.0])
    G = build_knn_graph(data, k=1)
    # Взаимная симметрия
    assert G.has_edge(0, 1)
    assert not G.has_edge(0, 2)
    # Степени
    degrees = dict(G.degree())
    assert degrees == {0: 1, 1: 2, 2: 1}


def test_build_distance_graph_threshold():
    data = np.array([0.0, 0.5, 2.0])
    G = build_distance_graph(data, d=1.0)
    # Узел 0 и 1 соединены, 2 изолирован
    assert G.has_edge(0, 1)
    assert not G.has_edge(0, 2) and not G.has_edge(1, 2)
