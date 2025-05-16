from scipy.stats import chi, chi2
import numpy as np


def generate_chi2(nu: int, n: int) -> np.ndarray:
    """Генерация данных из χ²"""
    return chi2.rvs(df=nu, size=n)


def generate_chi(nu: int, n: int) -> np.ndarray:
    """Генерация данных из χ"""
    return chi.rvs(df=nu, size=n)