"""
LoSanR experiment runner: sanitize private documents at chosen epsilon values
and evaluate semantic similarity, BLEU, ROUGE, and downstream classification F1.

Setup:
  - Train split is divided into public half (sentence pool + clusters) and private half (to sanitize)
  - Test split is held out for downstream F1 evaluation only
  - ALL train sentence embeddings and test doc embeddings are cached once per dataset
    (shared cache). Changing max_public/max_private/seed just selects different subsets
    from the same cache - no re-embedding needed.
  - Results are checkpointed after each epsilon so crashes don't lose work

Usage:
    python run_experiment.py --dataset imdb --epsilons 100 250 500 1000 --max_private 200
"""
import argparse
import json
import os
import time
import pickle
import numpy as np
from tqdm import tqdm

from src.common.embeddings import SentenceEmbedder, PCAProjector, normalize_embeddings as normalize
from src.common.clustering import SentenceClusterIndex
from src.common.data import load_dataset_splits, split_into_sentences
from src.common.evaluation import run_all_metrics, downstream_f1
from src.losanr.pipeline import LoSanRPipeline


def log(msg):
    print(msg, flush=True)


def get_shared_cache_dir(output_dir, dataset, model_name):
    """Shared cache: stores ALL train/test embeddings for a dataset+model. Param-independent."""
    # sanitize model name for filesystem (e.g. "all-MiniLM-L6-v2" -> "all-MiniLM-L6-v2")
    safe_model = model_name.replace("/", "_").replace("\\", "_")
    d = os.path.join(output_dir, f"{dataset}_{safe_model}_cache")
    os.makedirs(d, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# Shared caches (one-time per dataset, reused across all param combos)
# ---------------------------------------------------------------------------

def cache_train_sentences(train_texts, embedder, cache_dir):
    """Split ALL train docs into sentences, embed raw, cache as flat array + ranges."""
    sents_path = os.path.join(cache_dir, "train_doc_sentences.pkl")
    emb_path = os.path.join(cache_dir, "train_sent_embeddings.npy")
    ranges_path = os.path.join(cache_dir, "train_sent_ranges.pkl")

    if os.path.exists(sents_path) and os.path.exists(emb_path) and os.path.exists(ranges_path):
        log("  Loading cached train sentence data...")
        with open(sents_path, "rb") as f:
            all_doc_sentences = pickle.load(f)
        flat_embeddings = np.load(emb_path, mmap_mode='r')
        with open(ranges_path, "rb") as f:
            doc_sent_ranges = pickle.load(f)
        total = sum(len(s) for s in all_doc_sentences)
        log(f"  Loaded {total} sentences from {len(all_doc_sentences)} docs")
        return all_doc_sentences, flat_embeddings, doc_sent_ranges

    log("  Splitting all train docs into sentences...")
    all_doc_sentences = []
    flat_sentences = []
    doc_sent_ranges = []
    for doc in tqdm(train_texts, desc="  Splitting"):
        sents = split_into_sentences(doc)
        all_doc_sentences.append(sents)
        start = len(flat_sentences)
        flat_sentences.extend(sents)
        doc_sent_ranges.append((start, len(flat_sentences)))

    total = len(flat_sentences)
    log(f"  {total} sentences from {len(train_texts)} docs")
    log("  Embedding all train sentences (this may take a while)...")

    # Chunked + resumable embedding: write each chunk to disk so a killed
    # job can pick up where it left off. Chunks live next to the final file.
    CHUNK_SIZE = 10000
    chunks_dir = emb_path + ".chunks"
    os.makedirs(chunks_dir, exist_ok=True)
    n_chunks = (total + CHUNK_SIZE - 1) // CHUNK_SIZE
    for ci in range(n_chunks):
        chunk_path = os.path.join(chunks_dir, f"chunk_{ci:05d}.npy")
        if os.path.exists(chunk_path):
            continue
        start, end = ci * CHUNK_SIZE, min((ci + 1) * CHUNK_SIZE, total)
        log(f"  Embedding chunk {ci+1}/{n_chunks} (sentences {start}:{end})...")
        chunk_emb = embedder.embed(flat_sentences[start:end], normalize=False)
        # atomic write: write a tmp .npy then rename. np.save auto-appends
        # .npy so we use a tmp name ending in .npy to keep the filenames in
        # sync.
        tmp_path = chunk_path + ".tmp.npy"
        np.save(tmp_path, chunk_emb)
        os.replace(tmp_path, chunk_path)

    log("  Concatenating chunks...")
    flat_embeddings = np.concatenate(
        [np.load(os.path.join(chunks_dir, f"chunk_{ci:05d}.npy"))
         for ci in range(n_chunks)], axis=0)

    log("  Saving to shared cache...")
    with open(sents_path, "wb") as f:
        pickle.dump(all_doc_sentences, f)
    np.save(emb_path, flat_embeddings)
    with open(ranges_path, "wb") as f:
        pickle.dump(doc_sent_ranges, f)
    # Cleanup chunk files (final file is the source of truth now)
    import shutil
    shutil.rmtree(chunks_dir, ignore_errors=True)
    log("  Cached.")
    return all_doc_sentences, flat_embeddings, doc_sent_ranges


def cache_train_doc_embeddings(train_texts, embedder, cache_dir):
    """Embed ALL full train doc texts (raw). Normalize at runtime as needed."""
    path = os.path.join(cache_dir, "train_doc_embeddings_raw.npy")
    if os.path.exists(path):
        log("  Loading cached train doc embeddings...")
        return np.load(path, mmap_mode='r')
    log("  Embedding all train documents...")
    embs = embedder.embed(train_texts, normalize=False)
    np.save(path, embs)
    log(f"  Cached {len(train_texts)} doc embeddings")
    return embs


def cache_test_embeddings(test_texts, embedder, cache_dir):
    """Embed test docs (raw). Normalize at runtime as needed."""
    path = os.path.join(cache_dir, "test_embeddings_raw.npy")
    if os.path.exists(path):
        log("  Loading cached test embeddings...")
        return np.load(path, mmap_mode='r')
    log("  Embedding test documents...")
    embs = embedder.embed(test_texts, normalize=False)
    np.save(path, embs)
    log(f"  Cached {len(test_texts)} test embeddings")
    return embs


# ---------------------------------------------------------------------------
# Derived data (built at runtime from shared cache — cheap, no caching needed)
# ---------------------------------------------------------------------------

def build_pool(public_indices, all_doc_sentences, flat_sent_embeddings, doc_sent_ranges):
    """Build sentence pool from cached data by selecting public doc indices.
    Returns normalized pool embeddings (dot product = cosine sim)."""
    pool_sentences = []
    pool_emb_parts = []
    pool_doc_ranges = []

    for idx in public_indices:
        sents = all_doc_sentences[idx]
        start, end = doc_sent_ranges[idx]
        pool_start = len(pool_sentences)
        pool_sentences.extend(sents)
        pool_emb_parts.append(flat_sent_embeddings[start:end])
        pool_doc_ranges.append((pool_start, pool_start + len(sents)))

    pool_embeddings_raw = np.vstack(pool_emb_parts) if pool_emb_parts else np.empty((0, flat_sent_embeddings.shape[1]), dtype=np.float32)
    pool_embeddings = normalize(pool_embeddings_raw)

    log(f"  Pool: {len(pool_sentences)} sentences from {len(public_indices)} docs")
    return pool_sentences, pool_embeddings, pool_doc_ranges


def get_private_sentence_data(private_indices, all_doc_sentences, flat_sent_embeddings, doc_sent_ranges):
    """Extract per-doc sentence lists and normalized embeddings for private docs."""
    doc_sentences = []
    embeddings_norm = []

    for idx in private_indices:
        sents = all_doc_sentences[idx]
        start, end = doc_sent_ranges[idx]
        embs = flat_sent_embeddings[start:end]

        doc_sentences.append(sents)
        embeddings_norm.append(normalize(embs) if len(embs) > 0 else embs)

    return doc_sentences, embeddings_norm


# ---------------------------------------------------------------------------
# Clusters (cheap to build from pre-computed embeddings, no caching)
# ---------------------------------------------------------------------------

def build_clusters(pool_embeddings, n_clusters):
    log("  Building sentence-level clusters...")
    k = min(n_clusters, len(pool_embeddings))
    ci = SentenceClusterIndex(n_clusters=k)
    ci.fit(pool_embeddings)

    sizes = [len(ci.cluster_members[c]) for c in range(k)]
    log(f"  Clusters (K={k}): sizes min={min(sizes)}, max={max(sizes)}, mean={np.mean(sizes):.0f}")
    return ci


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

def run_pipeline_with_progress(pipeline, documents, name, doc_sentences=None, doc_embeddings=None):
    log(f"  Sanitizing with {name}...")
    results = []
    t0 = time.time()
    for idx, doc in enumerate(tqdm(documents, desc=f"  {name}")):
        sents = doc_sentences[idx] if doc_sentences is not None else None
        embs = doc_embeddings[idx] if doc_embeddings is not None else None
        results.append(pipeline.sanitize_document(doc, precomputed_sentences=sents, precomputed_embeddings=embs))
    elapsed = time.time() - t0
    log(f"  {name} done: {len(documents)} docs in {elapsed:.1f}s ({elapsed/len(documents):.2f}s/doc)")
    return results


def load_checkpoint(path):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {"results": {}}


def save_checkpoint(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="DP Text Sanitization Experiment")
    parser.add_argument("--dataset", type=str, default="imdb", choices=["atis", "snips", "imdb", "amazon", "drugscom", "agnews"])
    parser.add_argument("--epsilons", type=float, nargs="+", default=[100, 250, 500, 1000],
                        help="Document-level epsilon budgets (defaults match paper).")
    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument("--max_private", type=int, default=200,
                        help="Number of private documents to sanitize.")
    parser.add_argument("--max_public", type=int, default=None,
                        help="Cap on public docs used for sentence pool/PCA (None = all).")
    parser.add_argument("--public_ratio", type=float, default=0.5)
    parser.add_argument("--n_clusters", type=int, default=50)
    parser.add_argument("--top_k_clusters", type=int, default=3)
    parser.add_argument("--pca_dim", type=int, default=16,
                        help="PCA projection dim (default 16, matching the paper; 0 to disable PCA).")
    parser.add_argument("--chained_denoising", action="store_true",
                        help="Average each noisy embedding with immediate neighbors before retrieval.")
    parser.add_argument("--composition_mode", type=str, default="honest_doc",
                        choices=["per_sentence", "honest_doc"],
                        help="'honest_doc': epsilon is document-level budget, split across sentences "
                             "via basic composition (paper default). "
                             "'per_sentence': epsilon is per-sentence (no composition; reproduces the "
                             "inflated reporting style we critique in Section 2.5).")
    parser.add_argument("--model", type=str, default="all-MiniLM-L6-v2")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--skip_perplexity", action="store_true")
    parser.add_argument("--skip_bertscore", action="store_true")
    parser.add_argument("--skip_downstream", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_sentences", type=int, default=0,
                        help="If >0, only sanitize the first N sentences of each doc (0=all)")
    args = parser.parse_args()

    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # Checkpoint encodes run params so different configs don't collide
    pub_tag = f"pub{args.max_public}" if args.max_public else "pubAll"
    priv_tag = f"priv{args.max_private}" if args.max_private else "privAll"
    pca_tag = f"_pca{args.pca_dim}" if args.pca_dim > 0 else ""
    comp_tag = "_honest" if args.composition_mode == "honest_doc" else ""
    sent_tag = f"_sent{args.max_sentences}" if args.max_sentences > 0 else ""
    cd_tag = "_cd" if args.chained_denoising else ""
    checkpoint_path = os.path.join(args.output_dir, f"{args.dataset}_{pub_tag}_{priv_tag}_s{args.seed}{pca_tag}{comp_tag}{sent_tag}{cd_tag}_results.json")

    checkpoint = load_checkpoint(checkpoint_path)
    completed_epsilons = set(checkpoint["results"].keys())
    if completed_epsilons:
        log(f"Resuming: already completed epsilon(s) {completed_epsilons}")

    # --- Load dataset ---
    log(f"Loading dataset: {args.dataset}")
    data = load_dataset_splits(args.dataset)
    train_all = data["train"]
    test_data = data["test"]

    train_texts_all = [t for t, _ in train_all]
    train_labels_all = [l for _, l in train_all]
    test_texts = [t for t, _ in test_data]
    test_labels = [l for _, l in test_data]

    # --- Embedder ---
    log(f"\nInitializing embedder: {args.model}")
    embedder = SentenceEmbedder(args.model)

    # --- Build shared caches (once per dataset, reused across all param combos) ---
    shared_cache_dir = get_shared_cache_dir(args.output_dir, args.dataset, args.model)
    log("\nBuilding shared caches...")
    all_doc_sentences, flat_sent_embeddings, doc_sent_ranges = cache_train_sentences(train_texts_all, embedder, shared_cache_dir)
    all_doc_embeddings = cache_train_doc_embeddings(train_texts_all, embedder, shared_cache_dir)
    test_embeddings = cache_test_embeddings(test_texts, embedder, shared_cache_dir)

    # --- Split train into public/private (deterministic with seed) ---
    rng = np.random.RandomState(args.seed)
    indices = rng.permutation(len(train_all))
    split_idx = int(len(train_all) * args.public_ratio)
    public_indices = indices[:split_idx]
    private_indices = indices[split_idx:]

    if args.max_public:
        public_indices = public_indices[:args.max_public]
    if args.max_private:
        private_indices = private_indices[:args.max_private]

    private_texts = [train_texts_all[i] for i in private_indices]
    private_labels = [train_labels_all[i] for i in private_indices]

    log(f"\n  Public corpus: {len(public_indices)} docs")
    log(f"  Private data:  {len(private_indices)} docs")
    log(f"  Test set:      {len(test_texts)} docs")

    # --- Derive pool from shared cache (instant) ---
    log("\nBuilding sentence pool from cache...")
    pool_sentences, pool_embeddings, pool_doc_ranges = build_pool(
        public_indices, all_doc_sentences, flat_sent_embeddings, doc_sent_ranges)

    # --- Optional PCA projection (fit on public pool, project everything) ---
    pca = None
    if args.pca_dim > 0:
        log(f"\nFitting PCA (k={args.pca_dim}) on public pool...")
        pca = PCAProjector(n_components=args.pca_dim)
        pca.fit(pool_embeddings)
        pool_embeddings = pca.project(pool_embeddings)

    # --- Build clusters (fast from pre-computed embeddings) ---
    log("\nBuilding cluster index...")
    cluster_index = build_clusters(pool_embeddings, args.n_clusters)

    # --- Private data from shared cache (instant) ---
    orig_embeddings = normalize(all_doc_embeddings[private_indices])
    private_doc_sentences, private_sent_norm = get_private_sentence_data(
        private_indices, all_doc_sentences, flat_sent_embeddings, doc_sent_ranges)

    # Project private sentence embeddings if PCA is active
    if pca is not None:
        private_sent_norm = [pca.project(embs, normalize=True) if len(embs) > 0 else embs for embs in private_sent_norm]

    # Truncate to first N sentences if --max_sentences is set
    if args.max_sentences > 0:
        log(f"\n  Truncating to first {args.max_sentences} sentences per doc")
        private_doc_sentences = [s[:args.max_sentences] for s in private_doc_sentences]
        private_sent_norm = [e[:args.max_sentences] for e in private_sent_norm]

    total_priv_sents = sum(len(s) for s in private_doc_sentences)
    log(f"\n  Private: {total_priv_sents} sentences from {len(private_indices)} docs")

    # --- Baseline (no sanitization) ---
    if "baseline" not in checkpoint["results"]:
        log(f"\n{'='*60}")
        log("Computing baseline (no sanitization)")
        log(f"{'='*60}")
        baseline_metrics = run_all_metrics(
            original_texts=private_texts,
            sanitized_texts=private_texts,
            embedder=embedder,
            original_embeddings=orig_embeddings,
            sanitized_embeddings=orig_embeddings,
            skip_perplexity=args.skip_perplexity,
            skip_bertscore=args.skip_bertscore,
        )
        if not args.skip_downstream:
            log("  Baseline downstream F1...")
            ds_metrics = downstream_f1(
                train_texts=private_texts,
                train_labels=private_labels,
                test_texts=test_texts,
                test_labels=test_labels,
                embedder=embedder,
                test_embeddings=normalize(test_embeddings),
            )
            baseline_metrics.update(ds_metrics)
        checkpoint["results"]["baseline"] = baseline_metrics
        save_checkpoint(checkpoint_path, {"args": vars(args), "results": checkpoint["results"]})
        log(f"  Baseline: {json.dumps(baseline_metrics, indent=2)}")
    else:
        log(f"\n  Baseline already computed: {json.dumps(checkpoint['results']['baseline'], indent=2)}")

    # --- Run experiments ---
    all_results = checkpoint["results"]

    for eps in args.epsilons:
        eps_key = str(eps)
        if eps_key in completed_epsilons:
            log(f"\n  Skipping epsilon={eps} (already completed)")
            continue

        log(f"\n{'='*60}")
        log(f"Epsilon = {eps}")
        log(f"{'='*60}")

        eps_results = {}

        log(f"  Initializing LoSanR (composition_mode={args.composition_mode})...")
        losanr = LoSanRPipeline(
            embedder=embedder,
            cluster_index=cluster_index,
            pool_embeddings=pool_embeddings,
            pool_sentences=pool_sentences,
            epsilon=eps,
            delta=args.delta,
            top_k_clusters=args.top_k_clusters,
            composition_mode=args.composition_mode,
            use_chained_denoising=args.chained_denoising,
        )
        losanr_texts = run_pipeline_with_progress(losanr, private_texts, "LoSanR",
                                                  doc_sentences=private_doc_sentences,
                                                  doc_embeddings=private_sent_norm)
        pipelines_to_run = [("losanr", losanr_texts)]

        for name, sanitized in pipelines_to_run:
            log(f"\n  Evaluating {name}...")
            metrics = run_all_metrics(
                original_texts=private_texts,
                sanitized_texts=sanitized,
                embedder=embedder,
                original_embeddings=orig_embeddings,
                skip_perplexity=args.skip_perplexity,
                skip_bertscore=args.skip_bertscore,
            )

            if not args.skip_downstream:
                log(f"  Downstream F1 for {name}...")
                ds_metrics = downstream_f1(
                    train_texts=sanitized,
                    train_labels=private_labels,
                    test_texts=test_texts,
                    test_labels=test_labels,
                    embedder=embedder,
                    test_embeddings=normalize(test_embeddings),
                )
                metrics.update(ds_metrics)

            eps_results[name] = metrics
            log(f"  {name}: {json.dumps(metrics, indent=2)}")

        all_results[eps_key] = eps_results

        save_checkpoint(checkpoint_path, {"args": vars(args), "results": all_results})
        log(f"  Checkpoint saved after epsilon={eps}")

    log(f"\nAll done. Results at {checkpoint_path}")


if __name__ == "__main__":
    main()
