"""Linkage attacks on LoSanR with decoys matched on sentence count, including attackers
that use what the mechanism makes public.

LoSanR emits one replacement per input sentence, so sentence count alone identifies the
original among candidates of other lengths. Here every decoy has exactly the original's
sentence count, so the numbers measure what the content leaks beyond length.

Candidates come from the rest of the private half (never in the public pool):
  lookalike : the 3 same-length reviews nearest to the original (raw MiniLM, mean of sentences)
  random    : 3 random same-length reviews
  all       : up to 199 random same-length reviews (top-1 among up to 200 candidates)

Attackers (the candidate with the highest score is chosen):
  mean_raw    cosine of mean sentence embeddings, raw 384-d space (the attacker used so far)
  align_raw   sum over positions i of cos(replacement_i, candidate sentence_i), raw space
  align_pca   the same in the public PCA-16 space, where the noise was added
  align_ll    sum over positions of the log-density of the replacement's direction given the
              candidate sentence, under the mechanism's Gaussian noise with its known scale
              (replacement embedding used as a stand-in for the noisy point)

Uses numpy/scikit-learn only; no model is loaded. Writes results/attacks/matched_linkage.json
and results/attacks/inputs_matched.json (decoy texts for the LLM attacker).
"""
import json
import os
import pickle
import sys
from math import sqrt

import numpy as np
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.common.dp_mechanisms import _calibrate_analytic_gaussian  # noqa: E402

CACHE = "results/imdb_all-MiniLM-L6-v2_cache"
K, DELTA, SEED, MAX_SENTS, N_SHOTS = 16, 1e-5, 42, 15, 3


def unit_rows(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-10)


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def direction_loglik(sigma, d, grid):
    """log density (up to a constant) of the direction of x + sigma*z at cosine c from x,
    for unit x in R^d: exp(-(1-c^2)/(2 sigma^2)) * integral_0^inf r^(d-1) exp(-(r-c/sigma)^2/2) dr."""
    a = grid / sigma
    r = np.linspace(1e-6, max(a.max(), 0) + 14.0, 6000)
    lg = (d - 1) * np.log(r)[None, :] - 0.5 * (r[None, :] - a[:, None]) ** 2
    m = lg.max(axis=1, keepdims=True)
    integ = m[:, 0] + np.log(np.exp(lg - m).sum(axis=1) * (r[1] - r[0]))
    return -(1.0 - grid ** 2) / (2.0 * sigma ** 2) + integ


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--texts_dir", default="results/linkage/texts")
    ap.add_argument("--epsilons", type=int, nargs="+", default=[100, 250, 500, 1000])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--out", default="results/attacks/matched_linkage.json")
    args = ap.parse_args()
    with open(os.path.join(CACHE, "train_doc_sentences.pkl"), "rb") as f:
        doc_sents = pickle.load(f)
    flat = np.load(os.path.join(CACHE, "train_sent_embeddings.npy"), mmap_mode="r")
    with open(os.path.join(CACHE, "train_sent_ranges.pkl"), "rb") as f:
        ranges = pickle.load(f)

    n_train = len(doc_sents)
    perm = np.random.RandomState(SEED).permutation(n_train)
    half = int(n_train * 0.5)
    public, targets, rest = perm[:half], perm[half:half + 200], perm[half + 200:]

    def raw(idx):
        s, e = ranges[idx]
        return unit_rows(np.asarray(flat[s:e], dtype=np.float32))

    pool_sent, parts = [], []
    for idx in public:
        pool_sent.extend(doc_sents[idx])
        parts.append(raw(idx))
    pool_raw = np.vstack(parts)
    text2pool = {}
    for i, s in enumerate(pool_sent):
        text2pool.setdefault(s, i)
    pca = PCA(n_components=K, random_state=42).fit(pool_raw)
    proj = lambda X: unit_rows(pca.transform(X).astype(np.float32))
    pool_pca = proj(pool_raw)
    print(f"pool {len(pool_sent)} sentences; PCA-{K} explains {pca.explained_variance_ratio_.sum():.1%}")

    mean_of = lambda E: E.mean(axis=0) / max(np.linalg.norm(E.mean(axis=0)), 1e-10)
    rest_by_len = {}
    for idx in rest:
        rest_by_len.setdefault(len(doc_sents[idx]), []).append(int(idx))
    rest_mean = {int(idx): mean_of(raw(idx)) for idx in rest if len(doc_sents[idx]) > 0}

    rng = np.random.RandomState(SEED)
    plan, docs_out = {}, []
    for di, idx in enumerate(targets):
        n = len(doc_sents[idx])
        entry = {"doc_idx": di, "src_idx": int(idx), "n_sentences": n,
                 "original": " ".join(doc_sents[idx][:MAX_SENTS])}
        cands = rest_by_len.get(n, [])
        if n > 0 and len(cands) >= 3 and di >= N_SHOTS:
            o = mean_of(raw(idx))
            sims = np.array([float(o @ rest_mean[c]) for c in cands])
            look = [cands[j] for j in np.argsort(-sims)[:3]]
            rand = [cands[j] for j in rng.choice(len(cands), size=3, replace=False)]
            allc = [cands[j] for j in rng.choice(len(cands), size=min(199, len(cands)), replace=False)]
            plan[di] = {"lookalike": look, "random": rand, "all": allc,
                        "look_cos": float(np.sort(sims)[::-1][:3].mean()), "n_same_len": len(cands)}
            entry["lookalike_len"] = [" ".join(doc_sents[c][:MAX_SENTS]) for c in look]
            entry["random_len"] = [" ".join(doc_sents[c][:MAX_SENTS]) for c in rand]
        docs_out.append(entry)
    print(f"targets with at least 3 same-length candidates: {len(plan)} of {len(targets) - N_SHOTS}; "
          f"median same-length candidates {int(np.median([p['n_same_len'] for p in plan.values()]))}; "
          f"mean look-alike cosine {np.mean([p['look_cos'] for p in plan.values()]):.3f}")

    base = json.load(open("results/attacks/inputs.json", encoding="utf-8"))
    with open("results/attacks/inputs_matched.json", "w", encoding="utf-8") as f:
        json.dump({"meta": {**base["meta"], "matched_on": "sentence count"}, "docs": docs_out,
                   "conditions": base["conditions"]}, f)

    grid = np.linspace(-1.0, 1.0, 2001)
    sigma_cache, ll_cache, emb_cache = {}, {}, {}

    def embs(c):
        if c not in emb_cache:
            r = raw(c)
            emb_cache[c] = (r, proj(r))
        return emb_cache[c]
    results = {}
    names = ("mean_raw", "align_raw", "align_pca", "align_ll")
    kinds = ("lookalike", "random", "all")
    nT = len(plan)
    for eps in args.epsilons:
      per_seed = []
      for seed in args.seeds:
        rows = [json.loads(l) for l in open(
            os.path.join(args.texts_dir, f"losanr_pca16_ctx_eps{eps}_s{seed}.jsonl"), encoding="utf-8")]
        hits = {a: {k: 0 for k in kinds} for a in names}
        ks = []
        for di, p in plan.items():
            idx = int(targets[di])
            n = len(doc_sents[idx])
            ix = np.array([text2pool[s] for s in rows[di]["sanitized_sentences"]])
            assert len(ix) == n
            S_raw, S_pca = pool_raw[ix], pool_pca[ix]
            if (eps, n) not in ll_cache:
                if (eps, n) not in sigma_cache:
                    sigma_cache[(eps, n)] = _calibrate_analytic_gaussian(eps / n, DELTA, 2.0)
                ll_cache[(eps, n)] = direction_loglik(sigma_cache[(eps, n)], K, grid)
            ll = ll_cache[(eps, n)]

            def scores(c):
                C_raw, C_pca = embs(c)
                cos_pca = np.clip(np.sum(S_pca * C_pca, axis=1), -1, 1)
                return (float(mean_of(S_raw) @ mean_of(C_raw)),
                        float(np.sum(S_raw * C_raw)),
                        float(cos_pca.sum()),
                        float(np.interp(cos_pca, grid, ll).sum()))

            own = scores(idx)
            cache = {}
            for kind in ("lookalike", "random", "all"):
                dec = []
                for c in p[kind]:
                    if c not in cache:
                        cache[c] = scores(c)
                    dec.append(cache[c])
                for ai, a in enumerate(names):
                    hits[a][kind] += int(all(own[ai] > d[ai] for d in dec))
            ks.append(len(p["all"]) + 1)
        per_seed.append({a: {k: hits[a][k] / nT for k in kinds} for a in names})
      mean = {a: {k: float(np.mean([s[a][k] for s in per_seed])) for k in kinds} for a in names}
      std = {a: {k: float(np.std([s[a][k] for s in per_seed])) for k in kinds} for a in names}
      results[eps] = {"mean": mean, "std": std, "per_seed": per_seed, "n": nT, "seeds": list(args.seeds),
                      "median_candidates_all": int(np.median(ks))}
      print(f"\neps={eps}  (n={nT}, {len(args.seeds)} noise seed(s); chance 25% for 4-way; "
            f"'all' = top-1 among a median of {int(np.median(ks))} same-length candidates)")
      print(f"  {'attacker':10s} {'4-way look-alike':>18s} {'4-way random':>16s} {'top-1 among all':>17s}")
      for a in names:
          print(f"  {a:10s} " + " ".join(f"{mean[a][k]:10.3f} ± {std[a][k]:.3f}" for k in kinds))
    with open(args.out, "w") as f:
        json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
