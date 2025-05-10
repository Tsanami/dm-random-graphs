import numpy as np
import networkx as nx
from itertools import combinations

class GraphAnalyzer:
    def __init__(self, adjacency_matrix):
        """Инициализирует анализатор графов по матрице смежности."""
        self.G = nx.from_numpy_array(adjacency_matrix)
        self.n = adjacency_matrix.shape[0]

    def max_degree(self):
        """Возвращает максимальную степень вершины в графе."""
        return max(dict(self.G.degree()).values())

    def min_degree(self):
        """Возвращает минимальную степень вершины в графе."""
        return min(dict(self.G.degree()).values())

    def connected_components(self):
        """Вычисляет количество связных компонент графа."""
        return nx.number_connected_components(self.G)

    def articulation_points(self):
        """Возвращает количество точек сочленения в графе."""
        return len(list(nx.articulation_points(self.G)))

    def count_triangles(self):
        """Подсчитывает общее количество треугольников в графе."""
        return sum(nx.triangles(self.G).values()) // 3

    def chromatic_number(self):
        """
        Оценивает хроматическое число графа с помощью:
        1) жадного алгоритма (DSATUR)
        2) приближенных методов для небольших графов
        Возвращает минимум.
        """
        greedy_estimate = max(nx.coloring.greedy_color(self.G, strategy="DSATUR").values()) + 1

        approx_estimate = greedy_estimate
    
        # приближенный метод, если граф не большой
        if self.G.number_of_nodes() < 1000:
            approx_estimate = nx.algorithms.approximation.chromatic_number(self.G)
        return min(greedy_estimate, approx_estimate) # берем минимум из полученных результатов

    def clique_number(self):
        """Возвращает размер наибольшей клики в графе."""
        return nx.graph_clique_number(self.G)

    def max_independent_set(self, exact=False, warn_threshold=30):
        """
        Находит размер максимального независимого множества.
        Параметры:
            exact - если True, использует точный метод, но медленный
            warn_threshold - порог предупреждения для точного метода
        """
        if exact:
            if self.n > warn_threshold:
                print(f"[WARNING] Этот метод медленный для n > {warn_threshold}.")

            # Точный поиск: клика в дополнении ↔ независимое множество в оригинале
            comp = nx.complement(self.G)
            largest_clique = max(nx.find_cliques(comp), key=len)
            return len(largest_clique)

        else:
            # Быстрая аппроксимация
            approx_set = nx.algorithms.approximation.independent_set.maximum_independent_set(self.G)
            return len(approx_set)

    def dominating_number(self):
        """Возвращает размер доминирующего множества, найденного приближенным методом."""
        dominating_set = nx.algorithms.dominating_set(self.G)
        return len(dominating_set)

    def min_clique_cover(self):
        """
        Оценивает минимальное количество клик, необходимых для покрытия всех вершин графа.
        Реализация через раскраску дополнения графа (приближенно).
        """
        # Строим дополнение
        comp = nx.complement(self.G)
        # раскраска
        coloring = nx.greedy_color(comp, strategy='DSATUR')
        # Число цветов = размер покрытия кликами
        return max(coloring.values()) + 1 # получаем приближение