import numpy as np
from sklearn.cluster import MiniBatchKMeans


class SentenceClusterIndex:
    """Cluster individual sentences from the public sentence pool. Centroids are
    L2-normalized after fitting so that dot product equals cosine similarity."""

    def __init__(self, n_clusters: int = 50):
        self.n_clusters = n_clusters
        self.kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=4096, random_state=42)
        self.centroids: np.ndarray | None = None
        self.labels: np.ndarray | None = None
        self.cluster_members: dict[int, np.ndarray] = {}

    def fit(self, embeddings: np.ndarray):
        self.kmeans.fit(embeddings)
        centroids = self.kmeans.cluster_centers_.astype(np.float32)
        # normalize centroids so dot product == cosine similarity
        norms = np.linalg.norm(centroids, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        self.centroids = (centroids / norms).astype(np.float32)
        self.labels = self.kmeans.labels_
        for c in range(self.n_clusters):
            self.cluster_members[c] = np.where(self.labels == c)[0]
        return self

    def get_top_clusters(self, query: np.ndarray, top_k: int = 3) -> list[int]:
        sims = query @ self.centroids.T
        return np.argsort(sims)[-top_k:][::-1].tolist()

    def get_candidate_pool(self, query: np.ndarray, top_k: int = 3) -> np.ndarray:
        clusters = self.get_top_clusters(query, top_k)
        indices = np.concatenate([self.cluster_members[c] for c in clusters])
        return indices
