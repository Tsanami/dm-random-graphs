from sklearn.neighbors import NearestNeighbors
import numpy as np

def build_knn_graph(data, k):
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(data.reshape(-1, 1))
    adjacency = nbrs.kneighbors_graph(data.reshape(-1, 1)).toarray()
    np.fill_diagonal(adjacency, 0)
    return adjacency.astype(int)

def build_distance_graph(data, d):
    distances = np.abs(data[:, None] - data)
    adjacency = (distances <= d).astype(int)
    np.fill_diagonal(adjacency, 0)
    return adjacency

def build_knn_graph(data, k):
    """
    Строит симметричный KNN-граф: возвращает матрицу смежности.
    """
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(data.reshape(-1, 1))
    A = nbrs.kneighbors_graph(data.reshape(-1, 1)).toarray().astype(int)
    np.fill_diagonal(A, 0)
    # Делим итерации: делаем матрицу симметричной
    A = ((A + A.T) > 0).astype(int)
    return A


def build_distance_graph(data, d):
    """
    Строит граф пороговых расстояний: ребро, если |x_i - x_j| <= d.
    """
    distances = np.abs(data[:, None] - data)
    A = (distances <= d).astype(int)
    np.fill_diagonal(A, 0)
    return A
def build_knn_graph(data, k):
    """
    Строит симметричный KNN-граф: возвращает матрицу смежности.
    """
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(data.reshape(-1, 1))
    A = nbrs.kneighbors_graph(data.reshape(-1, 1)).toarray().astype(int)
    np.fill_diagonal(A, 0)
    # Оставляем только взаимные связи (intersection)
    A = (A & A.T)
    return A
