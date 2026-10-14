"""Sanitize the full private half of a dataset with one LoSanR variant, for BERT fine-tuning.

Same split (seed 42, second half of the train split) and same pipeline as run_experiment.py:
  --pca_dim K        0 = no PCA
  --retrieval        ctx (context-aware, the paper's default) or nn (plain nearest neighbor)
  --pool_tier        public corpus used for PCA fit, clustering and retrieval. Default: the
                     public half of --dataset. Other values: imdb, amazon, agnews (their public
                     halves) or rotten_tomatoes (text-only pool cache built by
                     analysis/corpus_tiers.py).
  --pool_size N      subsample the pool to N sentences (0 = full), same subsample as
                     analysis/corpus_tiers.py (seed 42)
All budgets in --epsilons are run in one process; the noise seed is reset before each one.
Requires the sentence cache written by run_experiment.py (results/{dataset}_{model}_cache).

Output: {out_dir}/{tag}_eps{E}_s{noise_seed}_sanitized.jsonl with {"label", "text"} rows.
Run from the repository root: python protocol/sanitize_variant.py --noise_seed 0 --tag imdb_k16_ctx
"""
import argparse
import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.common.embeddings import PCAProjector, normalize_embeddings  # noqa: E402
from src.common.clustering import SentenceClusterIndex  # noqa: E402
from src.common.data import load_dataset_splits  # noqa: E402
from src.losanr.pipeline import LoSanRPipeline  # noqa: E402

MODEL = "all-MiniLM-L6-v2"


def log(msg):
    print(msg, flush=True)


def load_cache(results_dir, dataset):
    d = os.path.join(results_dir, f"{dataset}_{MODEL}_cache")
    with open(os.path.join(d, "train_doc_sentences.pkl"), "rb") as f:
        sents = pickle.load(f)
    flat = np.load(os.path.join(d, "train_sent_embeddings.npy"), mmap_mode="r")
    with open(os.path.join(d, "train_sent_ranges.pkl"), "rb") as f:
        ranges = pickle.load(f)
    return sents, flat, ranges


def dataset_pool(results_dir, dataset, seed, public_ratio):
    sents, flat, ranges = load_cache(results_dir, dataset)
    perm = np.random.RandomState(seed).permutation(len(sents))
    pub = perm[: int(len(sents) * public_ratio)]
    pool, parts = [], []
    for idx in pub:
        s, e = ranges[idx]
        pool.extend(sents[idx])
        parts.append(flat[s:e])
    return pool, np.vstack(parts).astype(np.float32)


def corpus_pool(results_dir, corpus):
    d = os.path.join(results_dir, f"pool_{corpus}_{MODEL}_cache")
    with open(os.path.join(d, "sentences.pkl"), "rb") as f:
        pool = pickle.load(f)
    return pool, np.load(os.path.join(d, "sent_embeddings.npy")).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="imdb")
    ap.add_argument("--pool_tier", default=None)
    ap.add_argument("--pool_size", type=int, default=0)
    ap.add_argument("--pca_dim", type=int, default=16)
    ap.add_argument("--retrieval", choices=["ctx", "nn"], default="ctx")
    ap.add_argument("--epsilons", type=float, nargs="+", default=[100, 250, 500, 1000])
    ap.add_argument("--delta", type=float, default=1e-5)
    ap.add_argument("--noise_seed", type=int, required=True)
    ap.add_argument("--seed", type=int, default=42, help="split and pool-subsample seed")
    ap.add_argument("--public_ratio", type=float, default=0.5)
    ap.add_argument("--n_clusters", type=int, default=50)
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--out_dir", default="results/bert_eval/variants")
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    log(f"Loading dataset: {args.dataset}")
    train_all = load_dataset_splits(args.dataset)["train"]
    train_texts = [t for t, _ in train_all]
    train_labels = [l for _, l in train_all]
    doc_sents, flat, ranges = load_cache(args.results_dir, args.dataset)
    perm = np.random.RandomState(args.seed).permutation(len(train_texts))
    private_indices = perm[int(len(train_texts) * args.public_ratio):]
    log(f"  private docs: {len(private_indices)}")

    tier = args.pool_tier or args.dataset
    if tier in ("imdb", "amazon", "agnews"):
        pool_sentences, pool_raw = dataset_pool(args.results_dir, tier, args.seed, args.public_ratio)
    else:
        pool_sentences, pool_raw = corpus_pool(args.results_dir, tier)
    if 0 < args.pool_size < len(pool_sentences):
        idx = np.sort(np.random.RandomState(args.seed).choice(
            len(pool_sentences), size=args.pool_size, replace=False))
        pool_sentences, pool_raw = [pool_sentences[i] for i in idx], pool_raw[idx]
    pool_norm = normalize_embeddings(pool_raw)
    log(f"  pool: {tier}, {len(pool_sentences)} sentences")

    if args.pca_dim > 0:
        pca = PCAProjector(n_components=args.pca_dim)
        pca.fit(pool_norm)
        project = lambda e: pca.project(e, normalize=True)
    else:
        project = lambda e: e
    pool_embs = project(pool_norm)
    cluster_index = SentenceClusterIndex(n_clusters=min(args.n_clusters, len(pool_embs)))
    cluster_index.fit(pool_embs)

    for eps in args.epsilons:
        pipe = LoSanRPipeline(embedder=None, cluster_index=cluster_index, pool_embeddings=pool_embs,
                              pool_sentences=pool_sentences, epsilon=eps, delta=args.delta,
                              composition_mode="honest_doc")
        out = os.path.join(args.out_dir, f"{args.tag}_eps{eps:g}_s{args.noise_seed}_sanitized.jsonl")
        log(f"eps={eps:g} retrieval={args.retrieval} -> {out}")
        np.random.seed(args.noise_seed)
        with open(out + ".tmp", "w", encoding="utf-8") as f:
            for i, idx in enumerate(private_indices):
                sents = doc_sents[idx]
                s, e = ranges[idx]
                embs = project(normalize_embeddings(np.asarray(flat[s:e], dtype=np.float32))) if e > s else None
                if not sents:
                    text = train_texts[idx]  # same fallback as LoSanRPipeline.sanitize_document
                elif args.retrieval == "ctx":
                    text = pipe.sanitize_document(document=train_texts[idx], precomputed_sentences=sents,
                                                  precomputed_embeddings=embs)
                else:
                    mech = pipe._get_mechanism(len(sents))
                    noisy = np.stack([mech.privatize(x) for x in embs])
                    noisy /= np.maximum(np.linalg.norm(noisy, axis=1, keepdims=True), 1e-10)
                    best = np.argmax(noisy @ pool_embs.T, axis=1)
                    text = " ".join(pool_sentences[j] for j in best)
                f.write(json.dumps({"label": int(train_labels[idx]), "text": text}) + "\n")
                if (i + 1) % 2500 == 0:
                    log(f"  {i + 1}/{len(private_indices)}")
        os.replace(out + ".tmp", out)
    log("ALL_SANITIZE_DONE")


if __name__ == "__main__":
    sys.exit(main())
