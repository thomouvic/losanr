"""
Coherence of sanitized documents, plain nearest-neighbor vs context-aware retrieval.

Same data preparation, noise and retrieval as analysis/retrieval_ablation.py (200 private
documents, seed 42, PCA-16), but only the coherence metric: no classifier, no GPT-2.
Coherence is the mean cosine between consecutive replacement sentences in the PCA-16 space.
Used for the coherence table of the paper, including the tight budgets.

Usage (from the repository root):
    python analysis/retrieval_coherence.py --dataset imdb --delta 1e-5 --epsilons 25 50 100
"""
import argparse
import gc
import json
import os
import pickle
import sys

import numpy as np
from sklearn.cluster import KMeans

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis import retrieval_ablation as base  # noqa: E402
from src.common.embeddings import PCAProjector, normalize_embeddings as normalize  # noqa: E402
from src.common.data import load_dataset_splits  # noqa: E402
from src.common.dp_mechanisms import AnalyticGaussianMechanism  # noqa: E402

SEED = base.SEED
PCA_DIM = base.PCA_DIM


def mean_consecutive_coherence(per_doc_embs):
    """Mean over docs of (mean cos-sim between consecutive sentences); skips 1-sentence docs."""
    per_doc = []
    for embs in per_doc_embs:
        if len(embs) < 2:
            continue
        sims = np.sum(embs[:-1] * embs[1:], axis=1)
        per_doc.append(float(sims.mean()))
    return float(np.mean(per_doc)) if per_doc else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--delta", type=float, default=1e-5)
    ap.add_argument("--epsilons", type=float, nargs="+", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    np.random.seed(SEED)
    cache_dir = os.path.join(base.OUTPUT_DIR, f"{args.dataset}_{base.MODEL.replace('/', '_')}_cache")
    with open(os.path.join(cache_dir, "train_doc_sentences.pkl"), "rb") as f:
        all_doc_sentences = pickle.load(f)
    flat_sent_embeddings = np.load(os.path.join(cache_dir, "train_sent_embeddings.npy"), mmap_mode="r")
    with open(os.path.join(cache_dir, "train_sent_ranges.pkl"), "rb") as f:
        doc_sent_ranges = pickle.load(f)

    n_train = len(load_dataset_splits(args.dataset)["train"])
    rng = np.random.RandomState(SEED)
    indices = rng.permutation(n_train)
    split_idx = int(n_train * 0.5)
    public_indices = indices[:split_idx]
    private_indices = indices[split_idx:][:200]

    pool_idx_list = []
    for idx in public_indices:
        start, end = doc_sent_ranges[idx]
        pool_idx_list.extend(range(start, end))
    pool_embs_raw = normalize(np.array(flat_sent_embeddings[np.array(pool_idx_list, dtype=np.int64)], dtype=np.float32))
    base.log(f"pool: {len(pool_embs_raw)} sentences")

    private_sent_raw = []
    for idx in private_indices:
        start, end = doc_sent_ranges[idx]
        private_sent_raw.append(normalize(np.array(flat_sent_embeddings[start:end], dtype=np.float32)))
    del flat_sent_embeddings, all_doc_sentences
    gc.collect()

    pca = PCAProjector(n_components=PCA_DIM)
    pca.fit(pool_embs_raw)
    pool_embs = pca.project(pool_embs_raw, normalize=True)
    private_sent_pca = [pca.project(embs, normalize=True) for embs in private_sent_raw]
    del pool_embs_raw

    kmeans = KMeans(n_clusters=base.N_CLUSTERS, random_state=42, n_init=3)
    cluster_labels = kmeans.fit_predict(pool_embs)
    cluster_centers = normalize(kmeans.cluster_centers_.astype(np.float32))

    out = {"dataset": args.dataset, "delta": args.delta, "pca_dim": PCA_DIM,
           "n_private": len(private_sent_pca),
           "n_multi_sentence": sum(1 for e in private_sent_pca if len(e) >= 2),
           "original": mean_consecutive_coherence(private_sent_pca), "rows": []}
    base.log(f"original coherence: {out['original']:.4f}  (docs with 2+ sentences: {out['n_multi_sentence']})")
    base.log(f"{'eps':>6s}  {'plain NN':>9s}  {'context':>9s}")

    for eps in args.epsilons:
        row = {"eps": eps}
        for method in ["plain_nn", "context_aware"]:
            np.random.seed(SEED)
            per_doc = []
            for sent_embs in private_sent_pca:
                L = len(sent_embs)
                agm = AnalyticGaussianMechanism(epsilon=eps / L, delta=args.delta, sensitivity=2.0)
                noisy_embs = []
                for s in range(L):
                    noisy = agm.privatize(sent_embs[s])
                    noisy_embs.append(noisy / (np.linalg.norm(noisy) + 1e-10))
                noisy_embs = np.array(noisy_embs, dtype=np.float32)
                replacement_embs = []
                for s in range(L):
                    if method == "plain_nn":
                        best_idx = base.retrieve_plain_nn(noisy_embs[s], pool_embs)
                    else:
                        prev_emb = replacement_embs[-1] if s > 0 else None
                        best_idx = base.retrieve_context_aware(
                            s, noisy_embs, pool_embs, cluster_labels, cluster_centers,
                            prev_replacement_emb=prev_emb)
                    replacement_embs.append(pool_embs[best_idx])
                per_doc.append(np.array(replacement_embs, dtype=np.float32))
            row[method] = mean_consecutive_coherence(per_doc)
        out["rows"].append(row)
        base.log(f"{eps:>6g}  {row['plain_nn']:>9.4f}  {row['context_aware']:>9.4f}")

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(out, f, indent=1)
        base.log(f"wrote {args.out}")


if __name__ == "__main__":
    main()
