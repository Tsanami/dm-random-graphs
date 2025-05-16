import numpy as np
import networkx as nx
from src.build_graph import build_knn_graph, build_distance_graph


def test_build_knn_graph_small():
    data = np.array([0.0, 1.0, 2.0])
    A = build_knn_graph(data, k=1)
    # Матрица смежности должна быть симметричной и 3x3
    assert A.shape == (3, 3)
    assert np.allclose(A, A.T)
    # Каждый узел должен иметь ровно одну связь
    degrees = A.sum(axis=1)
    assert degrees.tolist() == [1, 1, 0]


def test_build_distance_graph_threshold():
    data = np.array([0.0, 0.5, 2.0])
    A = build_distance_graph(data, d=1.0)
    # Узел 0 и 1 соединены, 2 изолирован
    assert A[0,1] == 1 and A[1,0] == 1
    assert A[0,2] == 0 and A[1,2] == 0
