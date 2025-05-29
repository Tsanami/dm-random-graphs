import numpy as np
import networkx as nx
from src.graph_analyzer import GraphAnalyzer
from src.build_graph import build_distance_graph


def test_max_degree_and_chromatic_and_triangles():
    # Создаём граф: линия из трёх узлов
    G = nx.path_graph(3)
    ga = GraphAnalyzer(G)
    # Проверяем максимальную степень
    assert ga.max_degree() == 2
    # Хроматическое число для пути из 3 узлов = 2
    assert ga.chromatic_number() == 2
    # Число треугольников в пути = 0
    assert ga.count_triangles() == 0


def test_graph_analyzer_array_input_with_triangles():
    # Матрица смежности для пути из трёх узлов
    A = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    ga = GraphAnalyzer(A)
    # Проверяем максимальную степень
    assert ga.max_degree() == 2
    # Хроматическое число
    assert ga.chromatic_number() == 2
    # Число треугольников = 0
    assert ga.count_triangles() == 0


def test_complete_graph_triangles():
    # Полный граф на трех узлах (клика K3)
    Gk3 = nx.complete_graph(3)
    ga_k3 = GraphAnalyzer(Gk3)
    # В полном графе K3 каждая вершина степени 2
    assert ga_k3.max_degree() == 2
    # Хроматическое число для K3 = 3
    assert ga_k3.chromatic_number() == 3
    # В K3 ровно один треугольник
    assert ga_k3.count_triangles() == 1


def test_clique_number_with_threshold():
    # Узлы на координатах 0, 1, 3, d=1
    G = nx.Graph()
    coords = [0.0, 1.0, 3.0]
    for i, x in enumerate(coords):
        G.add_node(i, x=x)
    ga = GraphAnalyzer(G)
    # d=1: интервал ширины 1 может поймать точки [0,1] -> клика размер 2
    assert ga.clique_number(d=1.0) == 2
    # d=3: интервал ширины 3 покрывает все точки -> клика размер 3
    assert ga.clique_number(d=3.0) == 3


def test_clique_number_on_distance_graph():
    # Строим дистанционный граф из координат через build_distance_graph
    data = np.array([0.0, 1.0, 2.0, 5.0])
    d = 2.0
    G = build_distance_graph(data, d=d)
    ga = GraphAnalyzer(G)
    # Наибольшая клика должна соответствовать максимальному количеству точек
    # попадающих в интервал длины d: [0,2] -> 3 точки (0.0,1.0,2.0)
    assert ga.clique_number(d=d) == 3
