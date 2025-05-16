import numpy as np
from .build_graph import build_knn_graph, build_distance_graph
from .graph_analyzer import GraphAnalyzer


def monte_carlo_simulation(
    distribution,  # Функция генерации данных (generate_chi2/generate_chi)
    params: dict,  # Параметры распределения
    n_samples: int = 1000,
    graph_type: str = "knn",
    graph_param: float | int = 3,  # k для KNN, d для дистанционного
    metric: str = "count_triangles",  # Название характеристики из GraphAnalyzer
    metric_args: dict = None,  # Новый параметр для аргументов
) -> np.ndarray:
    """
    Выполняет симуляцию методом Монте-Карло для оценки распределения характеристики графа.

    Возвращает:
        Массив значений характеристики для всех симуляций.
    """
    if graph_type not in ["knn", "distance"]:
        raise ValueError(
            "Недопустимый тип графа. Используйте 'knn' или 'distance'."
        )

    T_values = []

    for _ in range(n_samples):
        # 1. Генерация данных
        data = distribution(**params)

        # 2. Построение графа
        if graph_type == "knn":
            G = build_knn_graph(data, k=graph_param)
        else:
            G = build_distance_graph(data, d=graph_param)

        # 3. Анализ характеристик
        analyzer = GraphAnalyzer(G)

        if not hasattr(analyzer, metric):
            raise ValueError(
                f"Метрика {metric} не существует в GraphAnalyzer."
            )

        if metric_args is None:
            metric_args = {}

        T = getattr(analyzer, metric)(
            **metric_args
        )  # Динамический выбор характеристики

        T_values.append(T)

    return np.array(T_values)
