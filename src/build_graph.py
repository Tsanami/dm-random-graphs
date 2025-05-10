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
