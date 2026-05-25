import numpy as np
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer


def normalize_embeddings(embs: np.ndarray) -> np.ndarray:
    """L2-normalize rows of an embedding matrix."""
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    return (embs / norms).astype(np.float32)


class PCAProjector:
    """Fit PCA on public pool embeddings, project any embeddings to k dimensions."""

    def __init__(self, n_components: int):
        self.n_components = n_components
        self.pca = PCA(n_components=n_components, random_state=42)

    def fit(self, embeddings: np.ndarray):
        self.pca.fit(embeddings)
        explained = self.pca.explained_variance_ratio_.sum()
        print(f"    [PCA] {self.n_components} components explain {explained:.1%} of variance", flush=True)
        return self

    def project(self, embeddings: np.ndarray, normalize: bool = True) -> np.ndarray:
        projected = self.pca.transform(embeddings).astype(np.float32)
        if normalize:
            projected = normalize_embeddings(projected)
        return projected


class SentenceEmbedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = None):
        self.model = SentenceTransformer(model_name, device=device)
        self.dim = self.model.get_sentence_embedding_dimension()

    def embed(self, sentences: list[str], normalize: bool = True, batch_size: int = 256) -> np.ndarray:
        vecs = self.model.encode(sentences, batch_size=batch_size, show_progress_bar=len(sentences) > 1000)
        if normalize:
            vecs = normalize_embeddings(vecs)
        return vecs.astype(np.float32)
