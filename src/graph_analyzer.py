import networkx as nx
import numpy as np


class GraphAnalyzer:
    def __init__(self, G: nx.Graph) -> None:
        """Инициализирует анализатор графов по матрице смежности."""
        if isinstance(G, np.ndarray):
            self.G = nx.from_numpy_array(G)
        else:
            self.G = G
        self.n = self.G.number_of_nodes()

    def max_degree(self) -> int:
        """Возвращает максимальную степень вершины в графе."""
        return max(dict(self.G.degree()).values())

    def min_degree(self) -> int:
        """Возвращает минимальную степень вершины в графе."""
        return min(dict(self.G.degree()).values())

    def connected_components(self) -> int:
        """Вычисляет количество связных компонент графа."""
        return nx.number_connected_components(self.G)

    def articulation_points(self) -> int:
        """Возвращает количество точек сочленения в графе."""
        return len(list(nx.articulation_points(self.G)))

    def count_triangles(self) -> int:
        """Подсчитывает общее количество треугольников в графе."""
        return sum(nx.triangles(self.G).values()) // 3

    def chromatic_number(self) -> int:
        """Жадное приближение хроматического числа (DSATUR)."""
        # Используем только greedy DSATUR
        colors = nx.coloring.greedy_color(self.G, strategy="DSATUR")
        return max(colors.values()) + 1

    def clique_number(self, d: float) -> int:
        """Возвращает размер наибольшей клики в графе.
        d - dist в дистанционном графе"""
        if self.G.number_of_nodes() == 0:
            raise ValueError("Граф пуст")

        # Извлекаем координаты из атрибутов узлов

        data = [self.G.nodes[node]["x"] for node in self.G.nodes]
        x = np.sort(data)

        max_clique = 0
        j = 0
        n = len(x)
        for i in range(n):
            while j < n and x[j] - x[i] <= d:
                j += 1
            max_clique = max(max_clique, j - i)
        return max_clique

    def max_independent_set(self, exact: bool = False, warn_threshold: int = 30) -> int:
        """
        Находит размер максимального независимого множества.
        Параметры:
            exact - если True, использует точный метод, но медленный
            warn_threshold - порог предупреждения для точного метода
        """
        if exact:
            if self.n > warn_threshold:
                print(f"[WARNING] Этот метод медленный для n > {warn_threshold}.")

            # Точный поиск: клика в дополнении ↔️ независимое множество в
            # оригинале
            comp = nx.complement(self.G)
            largest_clique = max(nx.find_cliques(comp), key=len)
            return len(largest_clique)

        else:
            # Быстрая аппроксимация
            approx_set = (
                nx.algorithms.approximation.independent_set.maximum_independent_set(
                    self.G
                )
            )
            return len(approx_set)

    def dominating_number(self) -> int:
        """Возвращает размер доминирующего множества, найденного приближенным методом."""
        dominating_set = nx.algorithms.dominating_set(self.G)
        return len(dominating_set)

    def min_clique_cover(self) -> int:
        """
        Оценивает минимальное количество клик, необходимых для покрытия всех вершин графа.
        Реализация через раскраску дополнения графа (приближенно).
        """
        # Строим дополнение
        comp = nx.complement(self.G)
        # раскраска
        coloring = nx.greedy_color(comp, strategy="DSATUR")
        # Число цветов = размер покрытия кликами
        return max(coloring.values()) + 1  # получаем приближение
