"""Linkage (distinguishability) ablation of LoSanR's two free lunches.

Grid: PCA dim (16, or 0 = no PCA) x retrieval (ctx = context-aware, nn = plain
nearest neighbor) x eps_doc x noise seed, on the 200 private docs used elsewhere
(split seed 42).

The attacker works in raw all-MiniLM-L6-v2 space (d=384) on mean-of-sentence
document embeddings (model-free re-identification, Appendix A.2 of the paper).
Every sanitized sentence is a pool sentence, so all embeddings come from the
cache and no model is loaded.

Per cell:
  - 200-way re-identification: top-1/5/10, mean rank, AUC
  - pairwise win rate against three decoys: random public doc, random public doc
    with the same label, and the public doc nearest to the original (hard)
  - empirical epsilon for k=2 (TeDA, Li et al. 2026, Eq. 12), point estimate and
    Clopper-Pearson 95% lower bound
  - first-n-sentences test: win rate when only the first n sentences are used
  - single-sentence win rate and mean per-sentence cosine to the original

Also writes the sanitized texts as JSONL ({out_dir}/texts/losanr_{pca}_{ctx|nn}_eps{E}_s{seed}.jsonl,
rows with doc_idx, src_idx, label, orig_text, sanitized_text, orig_sentences,
sanitized_sentences), which attacks/build_attack_inputs.py reads for the LLM attacks.

Usage (from the repository root):
  python attacks/linkage_ablation.py --cache_dir results/imdb_all-MiniLM-L6-v2_cache \
      --out_dir results/linkage --epsilons 25 50 100 250 500 1000
"""
import argparse
import json
import os
import pickle
import sys
from math import log as ln

import numpy as np
from scipy.stats import beta
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.common.embeddings import PCAProjector, normalize_embeddings  # noqa: E402
from src.common.clustering import SentenceClusterIndex  # noqa: E402
from src.common.data import load_dataset_splits  # noqa: E402
from src.losanr.pipeline import LoSanRPipeline  # noqa: E402


def log(msg):
    print(msg, flush=True)


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-10 else v


def mean_emb(E):
    """Mean of unit sentence embeddings, renormalized."""
    if len(E) == 0:
        return np.zeros(E.shape[1] if E.ndim == 2 else 384, dtype=np.float32)
    return unit(E.mean(axis=0)).astype(np.float32)


def reid_metrics(S, O):
    sim = S @ O.T
    n = sim.shape[0]
    true = np.diag(sim)
    ranks = (sim > true[:, None]).sum(axis=1) + 1
    labels = np.eye(n, dtype=int).ravel()
    return {
        "top1": float((ranks <= 1).mean()),
        "top5": float((ranks <= 5).mean()),
        "top10": float((ranks <= 10).mean()),
        "mean_rank": float(ranks.mean()),
        "auc": float(roc_auc_score(labels, sim.ravel())),
    }


def win_rate(S, O, D):
    t = np.sum(S * O, axis=1)
    d = np.sum(S * D, axis=1)
    return float((t > d).mean()), int((t > d).sum())


def eps_emp(wins, n, delta, alpha=0.05):
    """TeDA Eq. 12 for k=2: ln((p - delta) / (1 - p)), floored at 0."""
    def f(p):
        if p <= 0.5:
            return 0.0
        if p >= 1.0:
            return float("inf")
        return max(0.0, ln((p - delta) / (1.0 - p)))
    p_hat = wins / n
    p_lo = float(beta.ppf(alpha, wins, n - wins + 1)) if wins > 0 else 0.0
    return {"p": p_hat, "p_lower": p_lo, "eps_point": f(p_hat), "eps_lower": f(p_lo)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="imdb")
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--epsilons", type=float, nargs="+", default=[100, 250, 500, 1000])
    ap.add_argument("--pca_dims", type=int, nargs="+", default=[16, 0])
    ap.add_argument("--noise_seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--first_n", type=int, nargs="+", default=[1, 2, 3, 5, 8])
    ap.add_argument("--delta", type=float, default=1e-5)
    ap.add_argument("--split_seed", type=int, default=42)
    ap.add_argument("--public_ratio", type=float, default=0.5)
    ap.add_argument("--n_clusters", type=int, default=50)
    ap.add_argument("--max_private", type=int, default=200)
    args = ap.parse_args()

    text_dir = os.path.join(args.out_dir, "texts")
    os.makedirs(text_dir, exist_ok=True)

    log(f"Loading dataset: {args.dataset}")
    train_all = load_dataset_splits(args.dataset)["train"]
    train_texts = [t for t, _ in train_all]
    train_labels = np.array([l for _, l in train_all])

    with open(os.path.join(args.cache_dir, "train_doc_sentences.pkl"), "rb") as f:
        all_doc_sentences = pickle.load(f)
    flat = np.load(os.path.join(args.cache_dir, "train_sent_embeddings.npy"), mmap_mode="r")
    with open(os.path.join(args.cache_dir, "train_sent_ranges.pkl"), "rb") as f:
        ranges = pickle.load(f)

    rng = np.random.RandomState(args.split_seed)
    indices = rng.permutation(len(train_texts))
    split_idx = int(len(train_texts) * args.public_ratio)
    public_indices = indices[:split_idx]
    private_indices = indices[split_idx:split_idx + args.max_private]
    n_priv = len(private_indices)
    log(f"Public {len(public_indices)}, Private {n_priv}")

    def doc_embs(idx):
        s, e = ranges[idx]
        return normalize_embeddings(np.asarray(flat[s:e], dtype=np.float32)) if e > s \
            else np.zeros((0, flat.shape[1]), dtype=np.float32)

    # Public pool (raw, unit-normalized) and text -> pool index
    pool_sentences, parts = [], []
    for idx in public_indices:
        pool_sentences.extend(all_doc_sentences[idx])
        s, e = ranges[idx]
        parts.append(flat[s:e])
    pool_raw = normalize_embeddings(np.concatenate(parts, axis=0).astype(np.float32))
    text_to_pool = {}
    for i, s in enumerate(pool_sentences):
        text_to_pool.setdefault(s, i)
    log(f"Pool: {len(pool_sentences)} sentences")

    # Private originals (raw space)
    priv_embs = [doc_embs(idx) for idx in private_indices]
    priv_sents = [all_doc_sentences[idx] for idx in private_indices]
    O = np.stack([mean_emb(E) for E in priv_embs])
    n_sent_total = int(sum(len(E) for E in priv_embs))

    # Public doc embeddings for decoys
    pub_embs = [doc_embs(idx) for idx in public_indices]
    P = np.stack([mean_emb(E) for E in pub_embs])
    pub_labels = train_labels[public_indices]
    priv_labels = train_labels[private_indices]

    rng_d = np.random.RandomState(args.split_seed)
    dec_random = rng_d.choice(len(public_indices), size=n_priv, replace=False)
    dec_same = np.array([rng_d.choice(np.where(pub_labels == l)[0]) for l in priv_labels])
    dec_hard = np.argmax(O @ P.T, axis=1)
    D = {"random": P[dec_random], "same_label": P[dec_same], "hard": P[dec_hard]}
    log("Decoy cos to original (mean): " + ", ".join(
        f"{k}={float(np.sum(O * v, axis=1).mean()):.3f}" for k, v in D.items()))

    # Random pool sentence per private sentence, for the single-sentence test
    rng_s = np.random.RandomState(args.split_seed)
    sent_decoy_idx = rng_s.choice(len(pool_sentences), size=n_sent_total, replace=True)

    def evaluate(idx_lists):
        """idx_lists: per private doc, array of pool indices of the replacements."""
        S_sent = [pool_raw[ix] if len(ix) else np.zeros((0, pool_raw.shape[1]), np.float32)
                  for ix in idx_lists]
        S = np.stack([mean_emb(E) for E in S_sent])
        out = {"reid": reid_metrics(S, O), "decoy": {}}
        for name, Dm in D.items():
            wr, wins = win_rate(S, O, Dm)
            out["decoy"][name] = {"win_rate": wr, **eps_emp(wins, n_priv, args.delta)}
        out["doc_cos_true"] = float(np.sum(S * O, axis=1).mean())
        out["doc_cos_random_decoy"] = float(np.sum(S * D["random"], axis=1).mean())

        # first-n-sentences: sanitized, original and decoy all truncated to n
        out["first_n"] = {}
        for n in args.first_n:
            Sn = np.stack([mean_emb(E[:n]) for E in S_sent])
            On = np.stack([mean_emb(E[:n]) for E in priv_embs])
            Dn = np.stack([mean_emb(pub_embs[j][:n]) for j in dec_random])
            wr, _ = win_rate(Sn, On, Dn)
            out["first_n"][str(n)] = {"win_rate": wr, "top1": reid_metrics(Sn, On)["top1"]}

        # single sentence: each replacement vs its original vs a random pool sentence
        s_all = np.concatenate(S_sent, axis=0)
        o_all = np.concatenate(priv_embs, axis=0)
        d_all = pool_raw[sent_decoy_idx]
        t = np.sum(s_all * o_all, axis=1)
        d = np.sum(s_all * d_all, axis=1)
        out["sentence"] = {"win_rate": float((t > d).mean()), "mean_cos_true": float(t.mean()),
                           "mean_cos_decoy": float(d.mean()), "n": int(len(t))}
        flat_idx = np.concatenate([np.asarray(ix) for ix in idx_lists])
        out["unique_fraction"] = float(len(np.unique(flat_idx)) / max(1, len(flat_idx)))
        return out

    results = {"meta": vars(args), "cells": {}}

    # Floor: uniformly random replacement
    rng_rr = np.random.RandomState(args.split_seed)
    rr = [rng_rr.choice(len(pool_sentences), size=len(E), replace=False) for E in priv_embs]
    results["cells"]["random_replacement"] = evaluate(rr)
    log("random_replacement: " + json.dumps(results["cells"]["random_replacement"]["decoy"]["random"]))

    for k in args.pca_dims:
        tag = f"pca{k}" if k > 0 else "nopca"
        log(f"\n=== {tag} ===")
        if k > 0:
            pca = PCAProjector(n_components=k)
            pca.fit(pool_raw)
            pool_embs = pca.project(pool_raw, normalize=True)
            priv_proj = [pca.project(E, normalize=True) if len(E) else E for E in priv_embs]
        else:
            pool_embs = pool_raw
            priv_proj = priv_embs
        cluster_index = SentenceClusterIndex(n_clusters=args.n_clusters)
        cluster_index.fit(pool_embs)

        for eps in args.epsilons:
            pipe = LoSanRPipeline(
                embedder=None, cluster_index=cluster_index,
                pool_embeddings=pool_embs, pool_sentences=pool_sentences,
                epsilon=eps, delta=args.delta, composition_mode="honest_doc")
            for seed in args.noise_seeds:
                for mode in ("ctx", "nn"):
                    np.random.seed(seed)
                    idx_lists, rows = [], []
                    for di in range(n_priv):
                        sents, embs = priv_sents[di], priv_proj[di]
                        L = len(sents)
                        if L == 0:
                            ix = np.zeros(0, dtype=int)
                        elif mode == "ctx":
                            _, san = pipe.sanitize_document_paired(
                                document="", precomputed_sentences=sents,
                                precomputed_embeddings=embs)
                            ix = np.array([text_to_pool[s] for s in san])
                        else:
                            mech = pipe._get_mechanism(L)
                            ix = np.zeros(L, dtype=int)
                            for s in range(L):
                                noisy = unit(mech.privatize(embs[s]))
                                ix[s] = int(np.argmax(pool_embs @ noisy))
                        idx_lists.append(ix)
                        san_s = [pool_sentences[j] for j in ix]
                        src = int(private_indices[di])
                        rows.append({"doc_idx": di, "src_idx": src,
                                     "label": int(train_labels[src]),
                                     "orig_text": train_texts[src],
                                     "sanitized_text": " ".join(san_s),
                                     "orig_sentences": sents,
                                     "sanitized_sentences": san_s})
                    name = f"{tag}_{mode}_eps{int(eps)}_s{seed}"
                    with open(os.path.join(text_dir, f"losanr_{name}.jsonl"), "w",
                              encoding="utf-8") as f:
                        for r in rows:
                            f.write(json.dumps(r) + "\n")
                    res = evaluate(idx_lists)
                    res.update({"pca": tag, "retrieval": mode, "eps": eps, "seed": seed})
                    results["cells"][name] = res
                    dr = res["decoy"]
                    log(f"{name}: top1={res['reid']['top1']:.3f} auc={res['reid']['auc']:.3f} "
                        f"win rand={dr['random']['win_rate']:.3f} same={dr['same_label']['win_rate']:.3f} "
                        f"hard={dr['hard']['win_rate']:.3f} sent_cos={res['sentence']['mean_cos_true']:.3f}")
                    with open(os.path.join(args.out_dir, f"{args.dataset}_linkage_ablation.json"),
                              "w") as f:
                        json.dump(results, f, indent=1)

    # Summary: mean and std over noise seeds
    def agg(tag, mode, eps, getter):
        v = [getter(c) for c in results["cells"].values()
             if c.get("pca") == tag and c.get("retrieval") == mode and c.get("eps") == eps]
        return (float(np.mean(v)), float(np.std(v))) if v else (float("nan"), 0.0)

    md = [f"# Linkage ablation: {args.dataset}, {n_priv} private docs, "
          f"noise seeds {args.noise_seeds}", "",
          "Win rate = pairwise, sanitized doc closer to its original than to the decoy. "
          "eps_emp from the random-decoy win rate (k=2, point estimate).", "",
          "| PCA | Retrieval | eps | Top-1 | AUC | Win (random) | Win (same label) | Win (hard) "
          "| eps_emp | Sentence cos |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    getters = [
        lambda c: c["reid"]["top1"], lambda c: c["reid"]["auc"],
        lambda c: c["decoy"]["random"]["win_rate"], lambda c: c["decoy"]["same_label"]["win_rate"],
        lambda c: c["decoy"]["hard"]["win_rate"], lambda c: c["decoy"]["random"]["eps_point"],
        lambda c: c["sentence"]["mean_cos_true"]]
    for k in args.pca_dims:
        tag = f"pca{k}" if k > 0 else "nopca"
        for mode in ("ctx", "nn"):
            for eps in args.epsilons:
                cells = [f"{m:.3f} ± {s:.3f}" for m, s in (agg(tag, mode, eps, g) for g in getters)]
                md.append(f"| {tag} | {mode} | {int(eps)} | " + " | ".join(cells) + " |")
    md += ["", "## First-n-sentences win rate (random decoy), mean over seeds", "",
           "| PCA | Retrieval | eps | " + " | ".join(f"n={n}" for n in args.first_n) + " | all |",
           "|---|---|---|" + "---|" * (len(args.first_n) + 1)]
    for k in args.pca_dims:
        tag = f"pca{k}" if k > 0 else "nopca"
        for mode in ("ctx", "nn"):
            for eps in args.epsilons:
                vals = [agg(tag, mode, eps, lambda c, n=n: c["first_n"][str(n)]["win_rate"])[0]
                        for n in args.first_n]
                vals.append(agg(tag, mode, eps, lambda c: c["decoy"]["random"]["win_rate"])[0])
                md.append(f"| {tag} | {mode} | {int(eps)} | " + " | ".join(f"{v:.3f}" for v in vals) + " |")
    with open(os.path.join(args.out_dir, f"{args.dataset}_linkage_ablation.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    log("\n".join(md))
    log("Done.")


if __name__ == "__main__":
    sys.exit(main())
