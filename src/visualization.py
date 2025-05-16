import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Union
import numpy as np


def plot_distributions(
    h0_stats: np.ndarray,
    h1_stats: np.ndarray,
    metric_name: str,
    save_path: str = None,
) -> None:
    """
    Визуализирует распределения характеристик для H0 и H1.

    Параметры:
        h0_stats (np.ndarray): Статистика для гипотезы H0.
        h1_stats (np.ndarray): Статистика для гипотезы H1.
        metric_name (str): Название метрики для подписей.
        save_path (str): Путь для сохранения графика,
        если None, график отображается.
    """
    plt.figure(figsize=(10, 6))
    sns.kdeplot(h0_stats, label="H0", fill=True, alpha=0.5, color="blue")
    sns.kdeplot(h1_stats, label="H1", fill=True, alpha=0.5, color="red")
    plt.xlabel(metric_name, fontsize=12)
    plt.ylabel("Плотность", fontsize=12)
    plt.legend()
    plt.title(f"Распределение {metric_name} для H0 и H1", fontsize=14)
    plt.grid(True)
    _save_or_show(save_path)


def plot_line(
    x_values: List[Union[float, int]],
    y_values: List[float],
    x_label: str,
    y_label: str,
    title: str,
    color: str = "blue",
    save_path: str = None,
) -> None:
    """
    Строит линейный график зависимости y от x.

    Параметры:
        x_values (list): Значения по оси X.
        y_values (list): Значения по оси Y.
        x_label (str): Подпись оси X.
        y_label (str): Подпись оси Y.
        title (str): Заголовок графика.
        color (str): Цвет линии.
        save_path (str, optional): Путь для сохранения изображения,
        если None, покажет график.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(
        x_values,
        y_values,
        marker="o",
        linestyle="-",
        color=color,
        markersize=8,
    )
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def _save_or_show(save_path: str) -> None:
    """Вспомогательная функция для сохранения/отображения графика."""
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_critical_region(
    h0_stats: np.ndarray,
    h1_stats: np.ndarray,
    critical_value: float,
    title: str,
    xlabel: str,
    alpha: float = 0.05,
) -> None:
    """Визуализирует распределения H0/H1 и критическую область."""
    plt.figure(figsize=(10, 6))
    sns.kdeplot(h0_stats, label="H0", fill=True, color="blue")
    sns.kdeplot(h1_stats, label="H1", fill=True, color="orange")
    plt.axvline(
        critical_value,
        color="red",
        linestyle="--",
        label=f"Критерий (α={alpha})",
    )
    plt.xlabel(xlabel)
    plt.ylabel("Плотность")
    plt.title(title)
    plt.legend()
    plt.show()