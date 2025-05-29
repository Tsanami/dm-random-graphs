import numpy as np


def calculate_critical_region(h0_stats, alpha=0.05):
    """Вычисляет критическую область."""
    critical_value = np.quantile(h0_stats, 1 - alpha)
    return (-np.inf, critical_value)


def estimate_power(h1_stats, critical_value):
    """Оценивает мощность критерия."""
    return np.mean(h1_stats > critical_value)
