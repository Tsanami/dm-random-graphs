import numpy as np
import pytest
from src.data_utils import sample_stable, sample_normal


def test_sample_normal_shape_and_stats():
    sigma = 2.0
    n = 1000
    data = sample_normal(sigma, n)
    assert isinstance(data, np.ndarray)
    assert data.shape == (n,)
    # В среднем должно быть около нуля
    assert abs(np.mean(data)) < 0.2
    # Дисперсия около sigma^2
    assert abs(np.var(data) - sigma**2) < sigma**2 * 0.2


def test_sample_stable_shape():
    alpha = 1.5
    n = 500
    data = sample_stable(alpha, n)
    assert isinstance(data, np.ndarray)
    assert data.shape == (n,)
    # Проверяем, что значения не нулевые
    assert np.any(data != 0)