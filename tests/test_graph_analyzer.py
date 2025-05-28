import numpy as np
import networkx as nx
from src.graph_analyzer import GraphAnalyzer


def test_max_degree_and_chromatic_and_clique_and_triangles():
    # Создаём граф: линия из трёх узлов
    G = nx.path_graph(3)
    ga = GraphAnalyzer(G)
    # Проверяем максимальную степень
    assert ga.max_degree() == 2
    # Хроматическое число для пути из 3 узлов = 2
    assert ga.chromatic_number() == 2
    # Число треугольников в пути = 0
    assert ga.num_triangles() == 0
    # Кликовое число (размер наибольшей клики) для пути = 2
    assert ga.clique_number() == 2


def test_graph_analyzer_array_input_with_clique_and_triangles():
    # Матрица смежности для пути из трёх узлов
    A = np.array([[0, 1, 0],
                  [1, 0, 1],
                  [0, 1, 0]])
    ga = GraphAnalyzer(A)
    # Проверяем максимальную степень
    assert ga.max_degree() == 2
    # Хроматическое число
    assert ga.chromatic_number() == 2
    # Число треугольников = 0
    assert ga.num_triangles() == 0
    # Кликовое число = 2
    assert ga.clique_number() == 2


def test_complete_graph_triangles_and_clique():
    # Полный граф на трех узлах (клика K3)
    Gk3 = nx.complete_graph(3)
    ga_k3 = GraphAnalyzer(Gk3)
    # В полном графе K3 каждая вершина степени 2
    assert ga_k3.max_degree() == 2
    # Хроматическое число для K3 = 3
    assert ga_k3.chromatic_number() == 3
    # В K3 ровно один треугольник
    assert ga_k3.num_triangles() == 1
    # Кликовое число для K3 = 3
    assert ga_k3.clique_number() == 3
