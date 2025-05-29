import numpy as np
from src.test_statistics import T_knn, T_dist


def test_T_knn_simple():
    # Две точки, k=1 -> degree max = 1
    data = np.array([0.0, 1.0])
    val = T_knn(data, k=1)
    assert val == 1


def test_T_dist_simple():
    data = np.array([0.0, 2.0, 4.0])
    # при d<2 только изолированные
    val1 = T_dist(data, d=1.0)
    assert val1 == 1
    # при d>=2 все соединены -> clique size = 3
    val2 = T_dist(data, d=5.0)
    assert val2 == 3
