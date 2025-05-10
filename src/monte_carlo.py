import numpy as np
from .build_graph import build_knn_graph, build_distance_graph
from .graph_analyzer import GraphAnalyzer

def monte_carlo_simulation(
    distribution,  # Функция генерации данных (generate_h0/generate_h1)
    params: dict,   # Параметры распределения ({"mu":0, "sigma":1} и тп)
    n_samples: int = 1000,
    graph_type: str = "knn",
    graph_param: float | int = 3,  # k для KNN, d для дистанционного
    metric: str = "count_triangles"  # Название характеристики из GraphAnalyzer
) -> np.ndarray:
    """
    Выполняет симуляцию методом Монте-Карло для оценки распределения характеристики графа.
    
    Возвращает:
        Массив значений характеристики для всех симуляций.
    """
    T_values = []
    
    for _ in range(n_samples):
        # 1. Генерация данных
        data = distribution(**params)
        
        # 2. Построение графа
        if graph_type == "knn":
            adjacency = build_knn_graph(data, k=graph_param)
        else:
            adjacency = build_distance_graph(data, d=graph_param)
        
        # 3. Анализ характеристик
        analyzer = GraphAnalyzer(adjacency)
        T = getattr(analyzer, metric)()  # Динамический выбор характеристики
        
        T_values.append(T)
    
    return np.array(T_values)