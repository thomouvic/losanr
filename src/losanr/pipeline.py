"""
LoSanR pipeline: Analytic-Gaussian perturbation on unit-normalized embeddings + context-aware
nearest-neighbour retrieval from a sentence-clustered public pool.
"""
import numpy as np

from src.common.dp_mechanisms import AnalyticGaussianMechanism
from src.common.clustering import SentenceClusterIndex
from src.common.data import split_into_sentences


class LoSanRPipeline:
    def __init__(
        self,
        embedder,
        cluster_index: SentenceClusterIndex,
        pool_embeddings: np.ndarray,
        pool_sentences: list[str],
        epsilon: float,
        delta: float = 1e-5,
        top_k_clusters: int = 3,
        w_sim: float = 0.5,
        w_prev: float = 0.2,
        w_next: float = 0.2,
        w_cl: float = 0.1,
        composition_mode: str = "per_sentence",
        max_sentences: int = 60,
        use_chained_denoising: bool = False,
    ):
        self.embedder = embedder
        self.cluster_index = cluster_index
        self.pool_embeddings = pool_embeddings
        self.pool_sentences = pool_sentences
        self.top_k_clusters = top_k_clusters

        self.w_sim = w_sim
        self.w_prev = w_prev
        self.w_next = w_next
        self.w_cl = w_cl

        self.composition_mode = composition_mode
        self.epsilon = epsilon
        self.delta = delta
        self.use_chained_denoising = use_chained_denoising

        if composition_mode == "per_sentence":
            self.mechanism = AnalyticGaussianMechanism(epsilon=epsilon, delta=delta, sensitivity=2.0)
            self._mechanism_cache = None
        elif composition_mode == "honest_doc":
            self.mechanism = None
            self._mechanism_cache = {}
            for n in range(1, max_sentences + 1):
                eps_sent = epsilon / n
                self._mechanism_cache[n] = AnalyticGaussianMechanism(
                    epsilon=eps_sent, delta=delta, sensitivity=2.0
                )
        else:
            raise ValueError(f"Unknown composition_mode: {composition_mode}")

    def _get_mechanism(self, n_sentences: int) -> AnalyticGaussianMechanism:
        if self.composition_mode == "per_sentence":
            return self.mechanism
        if n_sentences in self._mechanism_cache:
            return self._mechanism_cache[n_sentences]
        # fallback for docs longer than max_sentences
        eps_sent = self.epsilon / n_sentences
        mech = AnalyticGaussianMechanism(epsilon=eps_sent, delta=self.delta, sensitivity=2.0)
        self._mechanism_cache[n_sentences] = mech
        return mech

    def _chain_denoise(self, noisy: np.ndarray) -> np.ndarray:
        """Average each noisy embedding with its immediate neighbors, then renormalize.
        Post-processing on already-privatized embeddings, so zero privacy cost."""
        n = len(noisy)
        denoised = np.empty_like(noisy)
        for i in range(n):
            acc = noisy[i].copy()
            count = 1
            if i > 0:
                acc += noisy[i - 1]
                count += 1
            if i < n - 1:
                acc += noisy[i + 1]
                count += 1
            denoised[i] = self._renormalize(acc / count)
        return denoised

    def _renormalize(self, x: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(x)
        if n < 1e-10:
            return x
        return x / n

    def sanitize_document(self, document: str, precomputed_sentences: list[str] = None, precomputed_embeddings: np.ndarray = None) -> str:
        sentences = precomputed_sentences if precomputed_sentences is not None else split_into_sentences(document)
        if not sentences:
            return document

        if precomputed_embeddings is not None:
            embeddings = precomputed_embeddings
        else:
            embeddings = self.embedder.embed(sentences, normalize=True)

        mechanism = self._get_mechanism(len(embeddings))
        noisy = np.array([self._renormalize(mechanism.privatize(e)) for e in embeddings])

        if self.use_chained_denoising:
            noisy = self._chain_denoise(noisy)

        # precompute weight normalization factors for the 3 cases:
        # first sentence (no prev), middle sentences (prev + next), last sentence (no next)
        n = len(noisy)
        w_base = self.w_sim + self.w_cl
        nf_first = 1.0 / (w_base + self.w_next) if n > 1 else 1.0 / w_base if w_base > 0 else 1.0
        nf_mid = 1.0 / (w_base + self.w_prev + self.w_next) if (w_base + self.w_prev + self.w_next) > 0 else 1.0
        nf_last = 1.0 / (w_base + self.w_prev) if (w_base + self.w_prev) > 0 else 1.0

        replacements = []
        prev_replacement_emb = None

        for i in range(n):
            hat_xi = noisy[i]

            candidate_indices = self.cluster_index.get_candidate_pool(hat_xi, self.top_k_clusters)
            if len(candidate_indices) == 0:
                candidate_indices = np.arange(len(self.pool_sentences))

            candidates = self.pool_embeddings[candidate_indices]

            has_prev = prev_replacement_emb is not None
            has_next = i + 1 < n
            if not has_prev:
                nf = nf_first
            elif not has_next:
                nf = nf_last
            else:
                nf = nf_mid

            scores = (self.w_sim * nf) * (candidates @ hat_xi)

            if has_prev:
                scores += (self.w_prev * nf) * (candidates @ prev_replacement_emb)

            if has_next:
                scores += (self.w_next * nf) * (candidates @ noisy[i + 1])

            top_cluster = self.cluster_index.get_top_clusters(hat_xi, 1)[0]
            candidate_labels = self.cluster_index.labels[candidate_indices]
            cluster_bonus = (candidate_labels == top_cluster).astype(np.float32)
            scores += (self.w_cl * nf) * cluster_bonus

            best_local = np.argmax(scores)
            best_idx = candidate_indices[best_local]
            replacements.append(self.pool_sentences[best_idx])
            prev_replacement_emb = self.pool_embeddings[best_idx]

        return " ".join(replacements)

    def sanitize_batch(self, documents: list[str]) -> list[str]:
        return [self.sanitize_document(doc) for doc in documents]

    def sanitize_document_paired(self, document: str, precomputed_sentences=None,
                                  precomputed_embeddings=None):
        """Like sanitize_document but returns (orig_sentences, replacements) pairs
        so callers can analyze per-position similarity (e.g., for reconstruction
        attack analysis). Same retrieval logic; only the return shape differs."""
        sentences = precomputed_sentences if precomputed_sentences is not None else split_into_sentences(document)
        if not sentences:
            return [], []

        if precomputed_embeddings is not None:
            embeddings = precomputed_embeddings
        else:
            embeddings = self.embedder.embed(sentences, normalize=True)

        mechanism = self._get_mechanism(len(embeddings))
        noisy = np.array([self._renormalize(mechanism.privatize(e)) for e in embeddings])
        if self.use_chained_denoising:
            noisy = self._chain_denoise(noisy)

        n = len(noisy)
        w_base = self.w_sim + self.w_cl
        nf_first = 1.0 / (w_base + self.w_next) if n > 1 else 1.0 / w_base if w_base > 0 else 1.0
        nf_mid = 1.0 / (w_base + self.w_prev + self.w_next) if (w_base + self.w_prev + self.w_next) > 0 else 1.0
        nf_last = 1.0 / (w_base + self.w_prev) if (w_base + self.w_prev) > 0 else 1.0

        replacements = []
        prev_replacement_emb = None
        for i in range(n):
            hat_xi = noisy[i]
            candidate_indices = self.cluster_index.get_candidate_pool(hat_xi, self.top_k_clusters)
            if len(candidate_indices) == 0:
                candidate_indices = np.arange(len(self.pool_sentences))
            candidates = self.pool_embeddings[candidate_indices]
            has_prev = prev_replacement_emb is not None
            has_next = i + 1 < n
            nf = nf_first if not has_prev else (nf_last if not has_next else nf_mid)
            scores = (self.w_sim * nf) * (candidates @ hat_xi)
            if has_prev:
                scores += (self.w_prev * nf) * (candidates @ prev_replacement_emb)
            if has_next:
                scores += (self.w_next * nf) * (candidates @ noisy[i + 1])
            top_cluster = self.cluster_index.get_top_clusters(hat_xi, 1)[0]
            candidate_labels = self.cluster_index.labels[candidate_indices]
            cluster_bonus = (candidate_labels == top_cluster).astype(np.float32)
            scores += (self.w_cl * nf) * cluster_bonus
            best_local = np.argmax(scores)
            best_idx = candidate_indices[best_local]
            replacements.append(self.pool_sentences[best_idx])
            prev_replacement_emb = self.pool_embeddings[best_idx]

        return sentences, replacements
