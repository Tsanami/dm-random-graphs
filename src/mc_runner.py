"""
Организация Monte Carlo экспериментов: генерация выборок, вычисление статистик через GraphAnalyzer, агрегация.
"""

import pandas as pd
from tqdm import tqdm
from src.data_utils import sample_stable, sample_normal
from src.test_statistics import T_knn, T_dist


def run_mc_experiment(
    dist: str, params: dict, graph_type: str, graph_param: float, n: int, n_iter: int
) -> pd.DataFrame:
    """
    Запускает MC-эксперимент для:
      - dist: 'stable' или 'normal'
      - params: {'alpha': ...} или {'sigma': ...}
      - graph_type: 'knn' или 'dist'
      - graph_param: k или d
      - n: размер выборки
      - n_iter: число повторений

    Возвращает DataFrame со столбцами: dist, параметр, graph_type, graph_param, n, stat, iter.
    """
    records = []
    for i in tqdm(range(n_iter), desc=f"MC {dist}-{graph_type}"):
        # Генерация данных
        data = (
            sample_stable(params["alpha"], n)
            if dist == "stable"
            else sample_normal(params["sigma"], n)
        )
        # Вычисление статистики
        if graph_type == "knn":
            stat = T_knn(data, graph_param)
        elif graph_type == "dist":
            stat = T_dist(data, graph_param)
        else:
            raise ValueError(f"Unknown graph_type: {graph_type}")
        # Сохранение результата
        record = {
            "dist": dist,
            **params,
            "graph_type": graph_type,
            "graph_param": graph_param,
            "n": n,
            "stat": stat,
            "iter": i,
        }
        records.append(record)
    return pd.DataFrame(records)


def aggregate_results(df: pd.DataFrame) -> pd.DataFrame:
    """
    Группирует результаты по (dist, graph_type, graph_param, n)
    и возвращает среднее и стандартное отклонение статистики.
    """
    return (
        df.groupby(["dist", "graph_type", "graph_param", "n"])
        .stat.agg(["mean", "std"])
        .reset_index()
    )
