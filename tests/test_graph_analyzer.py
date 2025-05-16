import numpy as np
import networkx as nx
from src.graph_analyzer import GraphAnalyzer


def test_max_degree_and_chromatic():
    # Создаём граф: линия из трёх узлов
    G = nx.path_graph(3)
    ga = GraphAnalyzer(G)
    assert ga.max_degree() == 2
    # Хроматическое число для пути из 3 узлов = 2
    assert ga.chromatic_number() == 2


def test_graph_analyzer_array_input():
    A = np.array([[0,1,0],[1,0,1],[0,1,0]])
    ga = GraphAnalyzer(A)
    assert ga.max_degree() == 2
    assert ga.chromatic_number() == 2
