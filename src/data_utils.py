# src/data_utils.py
"""
Утилиты для генерации выборок из распределений.
"""
import numpy as np


def sample_stable(alpha: float, n: int) -> np.ndarray:
    """
    Генерация n сэмплов из симметричного стабильного распределения Stable(alpha).
    Используем алгоритм Чамберса–Маллин–Стэпса.
    Параметры:
    ----------
    alpha : float
        Параметр устойчивости (0 < alpha <= 2).
    n : int
        Число выборок.
    Возвращает:
    -------
    samples : np.ndarray, shape (n,)
    """
    # Сэмплируем по формуле Чамберса–Маллин–Стэпса
    U = np.random.uniform(-np.pi / 2, np.pi / 2, size=n)
    W = np.random.exponential(scale=1.0, size=n)
    if alpha == 1.0:
        # Cauchy case
        return np.tan(U)
    else:
        numerator = np.sin(alpha * U)
        denominator = (np.cos(U)) ** (1 / alpha)
        frac = numerator / denominator
        factor = (np.cos(U - alpha * U) / W) ** ((1 - alpha) / alpha)
        return frac * factor


def sample_normal(sigma: float, n: int) -> np.ndarray:
    """
    Генерация n сэмплов из нормального распределения N(0, sigma^2).
    """
    return np.random.normal(loc=0.0, scale=sigma, size=n)
