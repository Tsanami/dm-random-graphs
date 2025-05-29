"""
Функции для проведения Monte Carlo экспериментов и критической области.
"""

import numpy as np
from .build_graph import build_knn_graph, build_distance_graph
from .graph_analyzer import GraphAnalyzer


def monte_carlo_simulation(
    distribution,  # функция генерации данных, например sample_stable или sample_normal
    params: dict,  # параметры распределения, ключи соответствуют аргументам distribution
    n_samples: int = 1000,
    graph_type: str = "knn",
    graph_param: float | int = 3,  # k для KNN, d для дистанционного
    metric: str = "max_degree",  # метод GraphAnalyzer: max_degree или chromatic_number
    metric_args: dict = None,  # дополнительные аргументы для метода
    n: int = 100,
) -> np.ndarray:
    """
    Выполняет Монте-Карло симуляцию для оценки распределения статистики графа.

    Параметры:
    ------------
    distribution : callable
        Функция генерации данных, возвращает массив размера n
    params : dict
        Аргументы для distribution
    n_samples : int
        Число повторений симуляции
    graph_type : {'knn','distance'}
    graph_param : float or int
    metric : {'max_degree','chromatic_number'}
    metric_args : dict
    n : размер выборки

    Возвращает:
    -------
    np.ndarray
        Массив значений статистики длины n_samples
    """
    if metric_args is None:
        metric_args = {}

    T = []
    for _ in range(n_samples):
        # генерируем данные
        data = distribution(**params)
        # создаем граф
        if graph_type == "knn":
            A = build_knn_graph(data.reshape(-1, 1), graph_param)
        elif graph_type == "distance":
            A = build_distance_graph(data, graph_param)
        else:
            raise ValueError("graph_type должен быть 'knn' или 'distance'")
        # анализируем
        ga = GraphAnalyzer(A)
        if not hasattr(ga, metric):
            raise ValueError(f"Метрика {metric} не найдена в GraphAnalyzer")
        stat = getattr(ga, metric)(**metric_args)
        T.append(stat)
    return np.array(T)

