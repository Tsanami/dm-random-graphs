import numpy as np
from src.monte_carlo import (
    monte_carlo_simulation,
    calculate_critical_region,
    estimate_power,
)
from src.distribution_generators import sample_normal


def test_monte_carlo_simulation_shape_and_values():
    # Используем normal для простоты
    stats = monte_carlo_simulation(
        sample_normal,
        {"sigma": 1.0, "n": 10},
        n_samples=100,
        graph_type="knn",
        graph_param=1,
        metric="max_degree",
    )
    assert isinstance(stats, np.ndarray)
    assert stats.shape == (100,)
    # Значения должны быть неотрицательными
    assert np.all(stats >= 0)


def test_critical_region_and_power():
    # Формируем простые данные
    h0 = np.zeros(100)
    h1 = np.ones(100)
    region, cv = calculate_critical_region(h0, alpha=0.05)
    # Критическое значение должно быть равно 0
    assert cv == 0
    # Мощность: h1 > 0 -> все True
    power = estimate_power(h1, cv)
    assert power == 1.0
