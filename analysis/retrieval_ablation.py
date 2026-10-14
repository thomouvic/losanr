"""
Retrieval ablation: context-aware scoring vs plain nearest-neighbor.

Both use the same PCA-16 sentence embeddings and DP noise.
- Plain NN: for each noisy embedding, pick the closest public sentence (full pool).
- Context-aware: use the scoring function with clustering (current pipeline).

Reports downstream F1 (logistic regression on 200 documents), adjacent-sentence coherence
and perplexity (GPT-2) of sanitized documents. Requires the IMDB cache written by
run_experiment.py. Run from the repository root: python analysis/retrieval_ablation.py
"""
import gc
import os
import pickle
import sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.cluster import KMeans

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.common.embeddings import PCAProjector, normalize_embeddings as normalize  # noqa: E402
from src.common.data import load_dataset_splits  # noqa: E402
from src.common.dp_mechanisms import AnalyticGaussianMechanism  # noqa: E402

SEED = 42
DATASET = "imdb"
MODEL = "all-MiniLM-L6-v2"
OUTPUT_DIR = "results"
PCA_DIM = 16
DELTA = 1e-5
EPSILONS = [100, 200, 500, 1000]
N_CLUSTERS = 50
TOP_K_CLUSTERS = 3

# Scoring weights (same as pipeline)
W_SIM = 0.5
W_PREV = 0.2
W_NEXT = 0.2
W_CL = 0.1


def log(msg):
    print(msg, flush=True)


def retrieve_plain_nn(noisy_emb, pool_embs):
    """Plain nearest-neighbor: highest cosine similarity."""
    sims = pool_embs @ noisy_emb
    return np.argmax(sims)


def retrieve_context_aware(i, noisy_embs, pool_embs, cluster_labels, cluster_centers,
                           prev_replacement_emb=None):
    """Context-aware retrieval with clustering and scoring."""
    hat_yi = noisy_embs[i]

    # Select top-k clusters
    cluster_sims = cluster_centers @ hat_yi
    top_clusters = np.argsort(cluster_sims)[-TOP_K_CLUSTERS:]
    candidate_mask = np.isin(cluster_labels, top_clusters)
    candidate_indices = np.where(candidate_mask)[0]
    candidates = pool_embs[candidate_indices]

    n = len(noisy_embs)
    has_prev = (i > 0) and (prev_replacement_emb is not None)
    has_next = (i < n - 1)

    # Determine which cluster is top-1
    top1_cluster = top_clusters[-1]
    in_top1 = (cluster_labels[candidate_indices] == top1_cluster).astype(np.float32)

    # Normalize weights based on available terms
    w_total = W_SIM + W_CL
    if has_prev:
        w_total += W_PREV
    if has_next:
        w_total += W_NEXT
    nf = 1.0 / w_total if w_total > 0 else 1.0

    scores = (W_SIM * nf) * (candidates @ hat_yi)
    scores += (W_CL * nf) * in_top1

    if has_prev:
        scores += (W_PREV * nf) * (candidates @ prev_replacement_emb)
    if has_next:
        scores += (W_NEXT * nf) * (candidates @ noisy_embs[i + 1])

    best_local = np.argmax(scores)
    return candidate_indices[best_local]


def compute_perplexity(documents, gpt2_model, gpt2_tokenizer, device="cpu"):
    """Compute mean perplexity of documents using GPT-2."""
    import torch
    perplexities = []
    for doc in documents:
        encodings = gpt2_tokenizer(doc, return_tensors="pt", truncation=True, max_length=1024)
        input_ids = encodings.input_ids.to(device)
        if input_ids.shape[1] < 2:
            continue
        with torch.no_grad():
            outputs = gpt2_model(input_ids, labels=input_ids)
            loss = outputs.loss.item()
        perplexities.append(np.exp(loss))
    return np.mean(perplexities) if perplexities else float("nan")


def main():
    np.random.seed(SEED)

    cache_dir = os.path.join(OUTPUT_DIR, f"{DATASET}_{MODEL.replace('/', '_')}_cache")
    log(f"Loading cached data from {cache_dir}...")

    with open(os.path.join(cache_dir, "train_doc_sentences.pkl"), "rb") as f:
        all_doc_sentences = pickle.load(f)
    flat_sent_embeddings = np.load(
        os.path.join(cache_dir, "train_sent_embeddings.npy"), mmap_mode='r')
    with open(os.path.join(cache_dir, "train_sent_ranges.pkl"), "rb") as f:
        doc_sent_ranges = pickle.load(f)

    data = load_dataset_splits(DATASET)
    train_all = data["train"]
    test_data = data["test"]
    train_labels_all = [l for _, l in train_all]
    test_labels = [l for _, l in test_data]

    # Split public/private
    rng = np.random.RandomState(SEED)
    indices = rng.permutation(len(train_all))
    split_idx = int(len(train_all) * 0.5)
    public_indices = indices[:split_idx]
    private_indices = indices[split_idx:][:200]
    private_labels = [train_labels_all[i] for i in private_indices]

    # Build pool: embeddings + text
    log("Building pool sentence embeddings and text...")
    pool_idx_list = []
    pool_sentences_text = []
    for idx in public_indices:
        start, end = doc_sent_ranges[idx]
        sents = all_doc_sentences[idx]
        for j_offset, j in enumerate(range(start, end)):
            pool_idx_list.append(j)
            if j_offset < len(sents):
                pool_sentences_text.append(sents[j_offset])
            else:
                pool_sentences_text.append("")

    pool_idx_arr = np.array(pool_idx_list, dtype=np.int64)
    pool_embs_raw = np.array(flat_sent_embeddings[pool_idx_arr], dtype=np.float32)
    pool_embs_raw = normalize(pool_embs_raw)
    log(f"  Pool: {len(pool_embs_raw)} sentences")

    # Build private sentence embeddings
    log("Building private sentence embeddings...")
    private_doc_sentences = []
    private_sent_raw = []
    for idx in private_indices:
        start, end = doc_sent_ranges[idx]
        sents = all_doc_sentences[idx]
        embs = np.array(flat_sent_embeddings[start:end], dtype=np.float32)
        embs = normalize(embs)
        private_doc_sentences.append(sents)
        private_sent_raw.append(embs)

    # Build test doc embeddings
    log("Loading test doc embeddings...")
    test_doc_embs_raw = np.array(
        np.load(os.path.join(cache_dir, "test_embeddings_raw.npy"), mmap_mode='r'),
        dtype=np.float32)
    test_doc_embs_raw = normalize(test_doc_embs_raw)

    # Free mmap
    del flat_sent_embeddings
    gc.collect()

    # PCA on pool (skip when PCA_DIM == 0; use raw normalized embeddings)
    if PCA_DIM > 0:
        log(f"Fitting PCA (k={PCA_DIM}) on pool...")
        pca = PCAProjector(n_components=PCA_DIM)
        pca.fit(pool_embs_raw)
        pool_embs = pca.project(pool_embs_raw, normalize=True)
        private_sent_pca = [pca.project(embs, normalize=True)
                            for embs in private_sent_raw]
        test_doc_embs = pca.project(test_doc_embs_raw, normalize=True)
    else:
        log("PCA_DIM=0; using raw 384-dim normalized embeddings.")
        pool_embs = normalize(pool_embs_raw)
        private_sent_pca = [normalize(embs) for embs in private_sent_raw]
        test_doc_embs = normalize(test_doc_embs_raw)
    del pool_embs_raw
    del test_doc_embs_raw

    # Cluster pool for context-aware retrieval
    log(f"Clustering pool into {N_CLUSTERS} clusters...")
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=3)
    cluster_labels = kmeans.fit_predict(pool_embs)
    cluster_centers = normalize(kmeans.cluster_centers_.astype(np.float32))

    # Load GPT-2 for perplexity
    log("Loading GPT-2 for perplexity...")
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    gpt2_tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")
    gpt2_model.eval()

    # Coherence baseline on ORIGINAL private docs in PCA space
    def mean_consecutive_coherence(per_doc_embs):
        """Mean over docs of (mean cos-sim between consecutive sentences).
        Skips docs with <2 sentences. Embeddings must be unit-normalized."""
        per_doc = []
        for embs in per_doc_embs:
            if len(embs) < 2:
                continue
            sims = np.sum(embs[:-1] * embs[1:], axis=1)
            per_doc.append(float(sims.mean()))
        return float(np.mean(per_doc)) if per_doc else float("nan")

    orig_coh = mean_consecutive_coherence(private_sent_pca)
    log(f"\nOriginal private-doc coherence (PCA-{PCA_DIM}): {orig_coh:.4f}")

    # Results header
    log(f"\n{'Eps':>6s}  {'F1 NN':>8s}  {'F1 Ctx':>8s}  {'dF1':>7s}  {'Coh NN':>8s}  {'Coh Ctx':>8s}  {'dCoh':>8s}  {'PPL NN':>8s}  {'PPL Ctx':>8s}  {'dPPL':>8s}")
    log(f"{'-'*100}")

    for eps in EPSILONS:
        results = {}
        for method in ["plain_nn", "context_aware"]:
            np.random.seed(SEED)

            sanitized_doc_embs = []
            sanitized_doc_texts = []
            replacement_embs_per_doc = []

            for doc_i in range(len(private_sent_pca)):
                sent_embs = private_sent_pca[doc_i]
                L = len(sent_embs)
                eps_s = eps / L
                agm = AnalyticGaussianMechanism(epsilon=eps_s, delta=DELTA, sensitivity=2.0)

                noisy_embs = []
                for s in range(L):
                    noisy = agm.privatize(sent_embs[s])
                    noisy = noisy / (np.linalg.norm(noisy) + 1e-10)
                    noisy_embs.append(noisy)
                noisy_embs = np.array(noisy_embs, dtype=np.float32)

                replacement_embs = []
                replacement_texts = []
                for s in range(L):
                    if method == "plain_nn":
                        best_idx = retrieve_plain_nn(noisy_embs[s], pool_embs)
                    else:
                        prev_emb = replacement_embs[-1] if s > 0 else None
                        best_idx = retrieve_context_aware(
                            s, noisy_embs, pool_embs, cluster_labels,
                            cluster_centers, prev_replacement_emb=prev_emb)
                    replacement_embs.append(pool_embs[best_idx])
                    replacement_texts.append(pool_sentences_text[best_idx])

                doc_emb = np.mean(replacement_embs, axis=0)
                doc_emb = doc_emb / (np.linalg.norm(doc_emb) + 1e-10)
                sanitized_doc_embs.append(doc_emb)
                sanitized_doc_texts.append(" ".join(replacement_texts))
                replacement_embs_per_doc.append(np.array(replacement_embs, dtype=np.float32))

            sanitized_doc_embs = np.array(sanitized_doc_embs, dtype=np.float32)

            clf = LogisticRegression(max_iter=1000, random_state=42)
            clf.fit(sanitized_doc_embs, private_labels)
            preds = clf.predict(test_doc_embs)
            f1 = f1_score(test_labels, preds, average="macro")

            ppl = compute_perplexity(sanitized_doc_texts, gpt2_model, gpt2_tokenizer)
            coh = mean_consecutive_coherence(replacement_embs_per_doc)

            results[method] = {"f1": f1, "ppl": ppl, "coh": coh}

        df1 = results["context_aware"]["f1"] - results["plain_nn"]["f1"]
        dppl = results["context_aware"]["ppl"] - results["plain_nn"]["ppl"]
        dcoh = results["context_aware"]["coh"] - results["plain_nn"]["coh"]
        log(f"{eps:>6d}  {results['plain_nn']['f1']:>8.4f}  {results['context_aware']['f1']:>8.4f}  {df1:>+7.4f}  "
            f"{results['plain_nn']['coh']:>8.4f}  {results['context_aware']['coh']:>8.4f}  {dcoh:>+8.4f}  "
            f"{results['plain_nn']['ppl']:>8.1f}  {results['context_aware']['ppl']:>8.1f}  {dppl:>+8.1f}")


if __name__ == "__main__":
    main()
