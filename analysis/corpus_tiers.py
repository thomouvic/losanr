"""
Public-corpus robustness: corpus tier (domain mismatch) x pool size grid.

How does LoSanR degrade when the public corpus is out-of-distribution, and how much
public data is needed? This is the 200-document, logistic-regression version; the
paper's main numbers for this question come from the BERT protocol
(protocol/sanitize_variant.py --pool_tier ... --pool_size ...), which reads the
pool caches that this script builds.

Design
  - Private set: IMDB, 200 docs, selected IDENTICALLY to run_experiment.py
    (seed-42 permutation, public_ratio=0.5, second half, first 200), so the
    in-distribution cell reproduces the k=16 row of the logistic-regression appendix.
  - Tiers (the public corpus replaces the IMDB pool for PCA fit, clustering,
    AND retrieval):
        imdb             in-distribution control
        rotten_tomatoes  similar: movie reviews, different source/register
        amazon           same genre (reviews), different topic
        agnews           different genre and topic (news)
        wikitext         encyclopedic (optional, off by default)
  - Pool sizes are counted in SENTENCES, not documents: sentences/doc differs
    ~7x across tiers (IMDB ~10.7, Amazon ~4.7, AG News ~1.5, RT 1), so
    doc-count matching would confound pool size with tier.
  - Per (tier, size): fit PCA-16 + clusters on the pool, sanitize with the
    paper pipeline (LoSanRPipeline, honest_doc composition), evaluate downstream
    LR F1 with the run_experiment.py protocol, over several noise seeds.
    A no-noise "ceiling" run separates pool-coverage loss from DP-noise loss.
  - Conditions (--conditions): "full" swaps the corpus for both PCA and
    retrieval (the deployment scenario). "pca_only" / "pool_only" attribute
    degradation to one role, with the other role served by the full
    in-distribution pool.

Privacy note: tiers change only public inputs, so the (eps, delta)-LDP
guarantee is identical in every cell. Mismatch can degrade utility, never privacy.

Results: results/corpus_tiers/<tier>.json (checkpointed, resume-safe).
Pool caches: results/pool_<corpus>_<model>_cache/ for the text-only corpora.

Usage (from the repository root):
    python analysis/corpus_tiers.py                                  # default grid
    python analysis/corpus_tiers.py --tiers rotten_tomatoes agnews --epsilons 100 500
    python analysis/corpus_tiers.py --conditions full pca_only pool_only --sizes 0
"""
import argparse
import json
import os
import pickle
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.common.embeddings import SentenceEmbedder, PCAProjector, normalize_embeddings as normalize  # noqa: E402
from src.common.clustering import SentenceClusterIndex  # noqa: E402
from src.common.data import load_dataset_splits, load_public_pool_texts, split_into_sentences  # noqa: E402
from src.common.evaluation import downstream_f1  # noqa: E402
from src.losanr.pipeline import LoSanRPipeline  # noqa: E402
from run_experiment import (  # noqa: E402
    get_shared_cache_dir,
    cache_train_sentences,
    cache_test_embeddings,
    get_private_sentence_data,
)

CONTROL_TIER = "imdb"

# Tier -> (pool source kind, human label used by make_corpus_tier_tables.py)
TIERS = {
    "imdb": ("dataset", "IMDB (in-distribution)"),
    "rotten_tomatoes": ("corpus", "Rotten Tomatoes (movie reviews)"),
    "amazon": ("dataset", "Amazon (product reviews)"),
    "agnews": ("dataset", "AG News (news)"),
    "wikitext": ("corpus", "Wikipedia (encyclopedic)"),
}
DEFAULT_TIERS = ["imdb", "rotten_tomatoes", "amazon", "agnews"]


def log(msg):
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# Pool construction (sentences + raw 384-d embeddings, cached per corpus)
# ---------------------------------------------------------------------------

def build_dataset_pool(dataset, embedder, output_dir, model_name, seed, public_ratio):
    """Pool from a labeled dataset's seed-`seed` PUBLIC half (same permutation as
    run_experiment.py), flattened to sentences. Reuses the shared per-dataset cache;
    the HF dataset is only loaded if the cache is missing."""
    cache_dir = get_shared_cache_dir(output_dir, dataset, model_name)
    sents_path = os.path.join(cache_dir, "train_doc_sentences.pkl")

    if os.path.exists(sents_path):
        log(f"  [{dataset}] loading pool from shared cache...")
        with open(sents_path, "rb") as f:
            all_doc_sentences = pickle.load(f)
        flat_embeddings = np.load(os.path.join(cache_dir, "train_sent_embeddings.npy"), mmap_mode="r")
        with open(os.path.join(cache_dir, "train_sent_ranges.pkl"), "rb") as f:
            doc_sent_ranges = pickle.load(f)
    else:
        log(f"  [{dataset}] shared cache missing; building it (one-time)...")
        data = load_dataset_splits(dataset)
        train_texts = [t for t, _ in data["train"]]
        all_doc_sentences, flat_embeddings, doc_sent_ranges = cache_train_sentences(
            train_texts, embedder, cache_dir)

    n_docs = len(all_doc_sentences)
    rng = np.random.RandomState(seed)
    indices = rng.permutation(n_docs)
    public_indices = indices[: int(n_docs * public_ratio)]

    sentences, emb_parts = [], []
    for idx in public_indices:
        start, end = doc_sent_ranges[idx]
        sentences.extend(all_doc_sentences[idx])
        emb_parts.append(flat_embeddings[start:end])
    raw = np.vstack(emb_parts).astype(np.float32)
    log(f"  [{dataset}] pool: {len(sentences)} sentences from {len(public_indices)} public docs")
    return sentences, raw


def build_corpus_pool(corpus, embedder, output_dir, model_name, max_docs=None):
    """Pool from a text-only public corpus (rotten_tomatoes, wikitext), flattened
    to sentences and cached under results/pool_<corpus>_<model>_cache/."""
    safe_model = model_name.replace("/", "_").replace("\\", "_")
    cache_dir = os.path.join(output_dir, f"pool_{corpus}_{safe_model}_cache")
    os.makedirs(cache_dir, exist_ok=True)
    sents_path = os.path.join(cache_dir, "sentences.pkl")
    emb_path = os.path.join(cache_dir, "sent_embeddings.npy")

    if os.path.exists(sents_path) and os.path.exists(emb_path):
        log(f"  [{corpus}] loading pool from cache...")
        with open(sents_path, "rb") as f:
            sentences = pickle.load(f)
        raw = np.load(emb_path).astype(np.float32)
    else:
        log(f"  [{corpus}] building pool (one-time download + embed)...")
        texts = load_public_pool_texts(corpus, max_docs=max_docs)
        sentences = []
        for t in texts:
            sentences.extend(split_into_sentences(t))
        raw = embedder.embed(sentences, normalize=False).astype(np.float32)
        with open(sents_path, "wb") as f:
            pickle.dump(sentences, f)
        np.save(emb_path, raw)
    log(f"  [{corpus}] pool: {len(sentences)} sentences")
    return sentences, raw


def build_tier_pool(tier, embedder, args):
    kind, _ = TIERS[tier]
    if kind == "dataset":
        return build_dataset_pool(tier, embedder, args.output_dir, args.model,
                                  args.seed, args.public_ratio)
    return build_corpus_pool(tier, embedder, args.output_dir, args.model,
                             max_docs=args.wikitext_max_docs if tier == "wikitext" else None)


def subsample_pool(sentences, raw, n_sentences, seed):
    """Uniform sentence-level subsample, fixed across noise seeds so noise seeds
    measure mechanism randomness only. Safe because the pipeline never uses
    pool document structure."""
    if n_sentences <= 0 or n_sentences >= len(sentences):
        return sentences, raw
    rng = np.random.RandomState(seed)
    idx = np.sort(rng.choice(len(sentences), size=n_sentences, replace=False))
    return [sentences[i] for i in idx], raw[idx]


# ---------------------------------------------------------------------------
# Private setup (identical selection to run_experiment.py)
# ---------------------------------------------------------------------------

def load_private_setup(args, embedder):
    data = load_dataset_splits(args.private_dataset)
    train_all = data["train"]
    test_data = data["test"]
    train_texts_all = [t for t, _ in train_all]
    train_labels_all = [l for _, l in train_all]
    test_texts = [t for t, _ in test_data]
    test_labels = [l for _, l in test_data]

    cache_dir = get_shared_cache_dir(args.output_dir, args.private_dataset, args.model)
    all_doc_sentences, flat_sent_embeddings, doc_sent_ranges = cache_train_sentences(
        train_texts_all, embedder, cache_dir)
    test_embeddings = cache_test_embeddings(test_texts, embedder, cache_dir)

    # Same permutation and slicing as run_experiment.py
    rng = np.random.RandomState(args.seed)
    indices = rng.permutation(len(train_all))
    split_idx = int(len(train_all) * args.public_ratio)
    private_indices = indices[split_idx:]
    if args.max_private:
        private_indices = private_indices[: args.max_private]

    doc_sentences, _, sent_norm384 = get_private_sentence_data(
        private_indices, all_doc_sentences, flat_sent_embeddings, doc_sent_ranges)

    n_sents = sum(len(s) for s in doc_sentences)
    log(f"  Private: {len(private_indices)} docs, {n_sents} sentences; "
        f"test: {len(test_texts)} docs")
    return {
        "texts": [train_texts_all[i] for i in private_indices],
        "labels": [train_labels_all[i] for i in private_indices],
        "doc_sentences": doc_sentences,
        "sent_norm384": sent_norm384,  # per-doc L2-normalized 384-d embeddings
        "test_texts": test_texts,
        "test_labels": test_labels,
        "test_emb_norm": normalize(np.asarray(test_embeddings, dtype=np.float32)),
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def diversity_stats(sent_lists):
    """How diverse is the sanitized output given a finite pool?"""
    all_repl = [s for doc in sent_lists for s in doc]
    counts = Counter(all_repl)
    docs = [" ".join(d) for d in sent_lists]
    n = max(len(all_repl), 1)
    return {
        "n_replacements": len(all_repl),
        "n_unique_sentences": len(counts),
        "unique_fraction": len(counts) / n,
        "max_sentence_reuse": max(counts.values()) if counts else 0,
        "n_identical_documents": len(docs) - len(set(docs)),
    }


def replacement_fidelity(sent_lists, sent_norm384, text2emb):
    """Mean cosine (raw 384-d space) between each original sentence and its
    replacement. Independent of the tier's PCA, so comparable across tiers."""
    sims = []
    for doc_repl, doc_orig in zip(sent_lists, sent_norm384):
        for i, r in enumerate(doc_repl):
            if i < len(doc_orig):
                sims.append(float(doc_orig[i] @ text2emb[r]))
    return float(np.mean(sims)) if sims else float("nan")


def evaluate_run(sent_lists, priv, embedder, text2emb):
    sanitized_texts = [" ".join(sl) if sl else "" for sl in sent_lists]
    ds = downstream_f1(
        train_texts=sanitized_texts,
        train_labels=priv["labels"],
        test_texts=priv["test_texts"],
        test_labels=priv["test_labels"],
        embedder=embedder,
        test_embeddings=priv["test_emb_norm"],
    )
    return {
        "f1_macro": ds["f1_macro"],
        "f1_micro": ds["f1_micro"],
        "fidelity_cos384": replacement_fidelity(sent_lists, priv["sent_norm384"], text2emb),
        "diversity": diversity_stats(sent_lists),
    }


def eps_key(eps: float) -> str:
    """Canonical checkpoint key: '100' not '100.0'."""
    return str(int(eps)) if float(eps).is_integer() else str(eps)


class _IdentityMechanism:
    """No-noise stand-in for the AGM: retrieval on clean projected embeddings.
    Gives the per-cell utility ceiling (pool-coverage loss only, no DP noise)."""
    sigma = 0.0

    def privatize(self, x):
        return x


# ---------------------------------------------------------------------------
# One grid cell = (pca source pool, retrieval pool) at fixed size/condition
# ---------------------------------------------------------------------------

def run_condition_cell(cell, pca_fit_raw, pool_sentences, pool_raw, priv, embedder,
                       args, save_fn):
    """Fill in missing (ceiling / eps / seed) entries of `cell` in place,
    calling save_fn(cell) after each completed unit for resume safety."""
    if args.pca_dim > 0:
        log(f"    Fitting PCA-{args.pca_dim} on {len(pca_fit_raw)} sentences; "
            f"retrieval pool {len(pool_sentences)} sentences")
        pca = PCAProjector(n_components=args.pca_dim)
        pca.fit(normalize(pca_fit_raw))
        project = lambda e: pca.project(e, normalize=True)
    else:
        # --pca_dim 0: no PCA, perturb and retrieve in the raw 384-d space
        log(f"    No PCA; retrieval pool {len(pool_sentences)} sentences")
        pca = None
        project = lambda e: e

    pool_norm384 = normalize(pool_raw)
    pool_proj = project(pool_norm384)
    clusters = SentenceClusterIndex(n_clusters=min(args.n_clusters, len(pool_proj)))
    clusters.fit(pool_proj)

    priv_proj = [project(e) if len(e) > 0 else e
                 for e in priv["sent_norm384"]]

    # First occurrence wins; duplicate pool texts share the same embedding.
    text2emb = {}
    for s, e in zip(pool_sentences, pool_norm384):
        text2emb.setdefault(s, e)

    # Subspace-mismatch diagnostics: how well the public PCA represents the
    # private sentences, in the raw 384-d space.
    if pca is not None:
        X = np.vstack([e for e in priv["sent_norm384"] if len(e) > 0])
        R = pca.pca.inverse_transform(pca.pca.transform(X))
        recon_cos = float(np.mean(np.sum(X * normalize(R), axis=1)))
        residual = float(np.mean(np.sum((X - R) ** 2, axis=1)))  # X rows are unit-norm
        cell["diagnostics"] = {
            "variance_explained_pool": float(pca.pca.explained_variance_ratio_.sum()),
            "recon_cos_private": recon_cos,
            "residual_frac_private": residual,
        }
    else:
        cell["diagnostics"] = {"variance_explained_pool": 1.0, "recon_cos_private": 1.0,
                               "residual_frac_private": 0.0}

    def sanitize_all(pipeline):
        return [
            pipeline.sanitize_document(
                priv["texts"][d],
                precomputed_sentences=priv["doc_sentences"][d],
                precomputed_embeddings=priv_proj[d],
                return_sentences=True,
            )
            for d in range(len(priv["texts"]))
        ]

    if "ceiling" not in cell and not args.skip_ceiling:
        log("    Ceiling (no-noise retrieval)...")
        ceiling_pipe = LoSanRPipeline(
            embedder=embedder, cluster_index=clusters,
            pool_embeddings=pool_proj, pool_sentences=pool_sentences,
            epsilon=1.0, delta=args.delta, top_k_clusters=args.top_k_clusters,
            composition_mode="per_sentence",
        )
        ceiling_pipe.mechanism = _IdentityMechanism()
        cell["ceiling"] = evaluate_run(sanitize_all(ceiling_pipe), priv, embedder, text2emb)
        log(f"    ceiling F1={cell['ceiling']['f1_macro']:.3f}")
        save_fn()

    cell.setdefault("eps", {})
    for eps in args.epsilons:
        eps_entry = cell["eps"].setdefault(eps_key(eps), {"seeds": {}})
        missing = [s for s in args.noise_seeds if str(s) not in eps_entry["seeds"]]
        if not missing:
            continue
        pipe = LoSanRPipeline(
            embedder=embedder, cluster_index=clusters,
            pool_embeddings=pool_proj, pool_sentences=pool_sentences,
            epsilon=eps, delta=args.delta, top_k_clusters=args.top_k_clusters,
            composition_mode="honest_doc",
        )
        for seed in missing:
            np.random.seed(seed)
            run = evaluate_run(sanitize_all(pipe), priv, embedder, text2emb)
            eps_entry["seeds"][str(seed)] = run
            log(f"    eps={eps} seed={seed}: F1={run['f1_macro']:.3f}")
        f1s = [v["f1_macro"] for v in eps_entry["seeds"].values()]
        eps_entry["f1_mean"] = float(np.mean(f1s))
        eps_entry["f1_std"] = float(np.std(f1s))
        save_fn()


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def tier_checkpoint_path(output_dir, tier):
    d = os.path.join(output_dir, "corpus_tiers")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{tier}.json")


def load_tier_checkpoint(path, tier, args):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {
        "meta": {
            "tier": tier,
            "tier_label": TIERS[tier][1],
            "model": args.model,
            "pca_dim": args.pca_dim,
            "delta": args.delta,
            "private_dataset": args.private_dataset,
            "max_private": args.max_private,
            "seed": args.seed,
            "n_clusters": args.n_clusters,
            "top_k_clusters": args.top_k_clusters,
        },
        "cells": {},
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Public-corpus tier x pool-size robustness grid")
    parser.add_argument("--tiers", nargs="+", default=DEFAULT_TIERS, choices=list(TIERS.keys()))
    parser.add_argument("--sizes", type=int, nargs="+", default=[2000, 10000, 50000, 0],
                        help="Pool sizes in SENTENCES; 0 = full pool")
    parser.add_argument("--epsilons", type=float, nargs="+", default=[100, 250, 500, 1000])
    parser.add_argument("--noise_seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--conditions", nargs="+", default=["full"],
                        choices=["full", "pca_only", "pool_only"],
                        help="full: tier corpus used for PCA+retrieval; pca_only/pool_only: "
                             "tier corpus in one role, full in-distribution pool in the other")
    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument("--pca_dim", type=int, default=16)
    parser.add_argument("--n_clusters", type=int, default=50)
    parser.add_argument("--top_k_clusters", type=int, default=3)
    parser.add_argument("--private_dataset", type=str, default="imdb")
    parser.add_argument("--max_private", type=int, default=200)
    parser.add_argument("--public_ratio", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42,
                        help="Split/subsample seed (must match run_experiment.py for comparability)")
    parser.add_argument("--model", type=str, default="all-MiniLM-L6-v2")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--skip_ceiling", action="store_true")
    parser.add_argument("--wikitext_max_docs", type=int, default=30000,
                        help="Paragraphs to stream from wikitext-103 (~4 sentences each)")
    args = parser.parse_args()

    log(f"Initializing embedder: {args.model}")
    embedder = SentenceEmbedder(args.model)

    log(f"\nLoading private setup ({args.private_dataset}, {args.max_private} docs)...")
    priv = load_private_setup(args, embedder)

    # Control pool (full, in-distribution) is needed for pca_only / pool_only.
    control_pool = None
    if any(c in args.conditions for c in ("pca_only", "pool_only")):
        log(f"\nLoading control pool ({CONTROL_TIER}, full) for decomposition conditions...")
        control_pool = build_tier_pool(CONTROL_TIER, embedder, args)

    for tier in args.tiers:
        log(f"\n{'=' * 60}\nTier: {tier} ({TIERS[tier][1]})\n{'=' * 60}")
        sentences_all, raw_all = build_tier_pool(tier, embedder, args)
        ckpt_path = tier_checkpoint_path(args.output_dir, tier)
        ckpt = load_tier_checkpoint(ckpt_path, tier, args)
        ckpt["meta"]["pool_sentences_total"] = len(sentences_all)

        for req in args.sizes:
            if req > 0 and req >= len(sentences_all):
                log(f"\n  size={req}: >= pool total ({len(sentences_all)}); covered by full, skipping")
                continue
            sents, raw = subsample_pool(sentences_all, raw_all, req, args.seed)
            size_label = "full" if req <= 0 else str(req)

            for cond in args.conditions:
                if tier == CONTROL_TIER and cond != "full":
                    continue  # decomposition is a no-op on the control tier
                key = f"{cond}|pool={size_label}"
                cell = ckpt["cells"].setdefault(key, {})
                cell["pool_size_requested"] = req
                cell["pool_size_actual"] = len(sents)
                cell["condition"] = cond
                log(f"\n  Cell {tier}/{key} ({len(sents)} sentences)")

                if cond == "full":
                    pca_raw, pool_s, pool_r = raw, sents, raw
                elif cond == "pca_only":
                    pca_raw, (pool_s, pool_r) = raw, control_pool
                else:  # pool_only
                    pca_raw, pool_s, pool_r = control_pool[1], sents, raw

                def save_fn():
                    with open(ckpt_path, "w") as f:
                        json.dump(ckpt, f, indent=2)

                run_condition_cell(cell, pca_raw, pool_s, pool_r, priv, embedder, args, save_fn)

        log(f"\n  Tier {tier} done. Results at {ckpt_path}")

    log("\nAll tiers done.")


if __name__ == "__main__":
    main()
