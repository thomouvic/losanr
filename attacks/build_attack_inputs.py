"""Build the inputs for the LLM attacks (reconstruction and look-alike linkage).

Targets: the 200 IMDB private documents used everywhere else (split seed 42).
Decoys:  drawn from the REST of the private half (never used as public pool, never
         sanitized), so the candidates carry no public-corpus bias:
           lookalike = the 3 nearest documents to the original (raw MiniLM, mean of sentences)
           random    = 3 uniformly random documents
Texts are rebuilt from the cached sentence split and capped at MAX_SENTS sentences so
that all candidates share one format.

Conditions (sanitized versions of the 200 targets):
  losanr_eps{E}      PCA-16, context-aware retrieval, noise seed 0, read from
                     {losanr_dir}/losanr_pca16_ctx_eps{E}_s0.jsonl (attacks/linkage_ablation.py)
  dpbart_eps{E}      DP-BART PR+ rewrites of the IMDB train split, read from
                     {dpbart_dir}/dp_bart_eps{E}.csv (rows "label,text" in train order)
Extra conditions can be added with --extra NAME=path.jsonl (rows with doc_idx and
sanitized_text), e.g. the SanText output of attacks/santext_baseline.py.

Output: results/attacks/inputs.json. Uses numpy only; no model is loaded.
"""
import argparse
import csv
import json
import os
import pickle

import numpy as np

MAX_SENTS = 15
N_DECOYS = 3


def unit_rows(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, 1e-10)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", default="results/imdb_all-MiniLM-L6-v2_cache")
    ap.add_argument("--losanr_dir", default="results/linkage/texts")
    ap.add_argument("--dpbart_dir", default="results/readability")
    ap.add_argument("--out", default="results/attacks/inputs.json")
    ap.add_argument("--extra", nargs="*", default=[])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_private", type=int, default=200)
    args = ap.parse_args()

    with open(os.path.join(args.cache_dir, "train_doc_sentences.pkl"), "rb") as f:
        doc_sents = pickle.load(f)
    flat = np.load(os.path.join(args.cache_dir, "train_sent_embeddings.npy"), mmap_mode="r")
    with open(os.path.join(args.cache_dir, "train_sent_ranges.pkl"), "rb") as f:
        ranges = pickle.load(f)

    n_train = len(doc_sents)
    perm = np.random.RandomState(args.seed).permutation(n_train)
    split = int(n_train * 0.5)
    targets = perm[split:split + args.max_private]
    rest = perm[split + args.max_private:]
    print(f"targets {len(targets)}, decoy candidates {len(rest)}")

    def doc_emb(idx):
        s, e = ranges[idx]
        if e <= s:
            return np.zeros(flat.shape[1], dtype=np.float32)
        m = unit_rows(np.asarray(flat[s:e], dtype=np.float32)).mean(axis=0)
        return m / max(np.linalg.norm(m), 1e-10)

    def text_of(idx):
        return " ".join(doc_sents[idx][:MAX_SENTS])

    T = np.stack([doc_emb(i) for i in targets])
    R = np.stack([doc_emb(i) for i in rest])
    sims = T @ R.T
    rng = np.random.RandomState(args.seed)

    docs = []
    for di, idx in enumerate(targets):
        near = np.argsort(-sims[di])[:N_DECOYS]
        rand = rng.choice(len(rest), size=N_DECOYS, replace=False)
        docs.append({
            "doc_idx": di, "src_idx": int(idx),
            "n_sentences": len(doc_sents[idx]),
            "original": text_of(idx),
            "lookalike_idx": [int(rest[j]) for j in near],
            "random_idx": [int(rest[j]) for j in rand],
            "lookalike": [text_of(rest[j]) for j in near],
            "lookalike_cos": [float(sims[di, j]) for j in near],
            "random": [text_of(rest[j]) for j in rand],
            "random_cos": [float(sims[di, j]) for j in rand],
        })
    print("mean cos to original: lookalike %.3f, random %.3f" % (
        np.mean([d["lookalike_cos"] for d in docs]), np.mean([d["random_cos"] for d in docs])))

    # Decoys with the original's sentence count (same selection as matched_linkage.py), so
    # that length cannot give the original away. Stored as lookalike_len / random_len.
    rest_by_len = {}
    for j, idx in enumerate(rest):
        rest_by_len.setdefault(len(doc_sents[idx]), []).append(j)
    rng2 = np.random.RandomState(args.seed)
    n_len = 0
    for di, idx in enumerate(targets):
        n = len(doc_sents[idx])
        cands = rest_by_len.get(n, [])
        if n == 0 or len(cands) < 3 or di < 3:
            continue
        s = np.array([sims[di, j] for j in cands])
        look = [cands[j] for j in np.argsort(-s)[:N_DECOYS]]
        rand = [cands[j] for j in rng2.choice(len(cands), size=N_DECOYS, replace=False)]
        rng2.choice(len(cands), size=min(199, len(cands)), replace=False)  # keep the stream aligned
        docs[di]["lookalike_len"] = [text_of(rest[j]) for j in look]
        docs[di]["lookalike_len_idx"] = [int(rest[j]) for j in look]
        docs[di]["random_len"] = [text_of(rest[j]) for j in rand]
        docs[di]["random_len_idx"] = [int(rest[j]) for j in rand]
        n_len += 1
    print(f"targets with same-length decoys: {n_len}")

    conditions = {}
    for eps in (10, 25, 50, 100, 250, 500, 1000):
        p = os.path.join(args.losanr_dir, f"losanr_pca16_ctx_eps{eps}_s0.jsonl")
        if not os.path.exists(p):
            continue
        rows = [json.loads(l) for l in open(p, encoding="utf-8")]
        assert [r["src_idx"] for r in rows] == [int(i) for i in targets], "LoSanR rows misaligned"
        conditions[f"losanr_eps{eps}"] = [" ".join(r["sanitized_sentences"][:MAX_SENTS]) for r in rows]

    csv.field_size_limit(10 ** 9)
    for eps in (250, 500, 1000):
        p = os.path.join(args.dpbart_dir, f"dp_bart_eps{eps}.csv")
        if not os.path.exists(p):
            continue
        rows = list(csv.reader(open(p, encoding="utf-8", errors="ignore")))
        assert len(rows) == n_train, f"DP-BART rows {len(rows)} != train {n_train}"
        conditions[f"dpbart_eps{eps}"] = [" ".join(rows[int(i)][1].split()) for i in targets]

    for spec in args.extra:
        name, path = spec.split("=", 1)
        rows = {json.loads(l)["doc_idx"]: json.loads(l) for l in open(path, encoding="utf-8")}
        conditions[name] = [rows[d]["sanitized_text"] for d in range(len(targets))]

    for name, texts in conditions.items():
        w = [len(t.split()) for t in texts]
        print(f"{name}: {len(texts)} docs, mean {np.mean(w):.0f} words, empty {sum(1 for x in w if x == 0)}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"meta": {"max_sents": MAX_SENTS, "n_decoys": N_DECOYS, "seed": args.seed},
                   "docs": docs, "conditions": conditions}, f)
    print("wrote", args.out)

    # DP-BART alignment check: do rewrites share more words with their own source than
    # with another document's opening?
    def overlap(a, b):
        a, b = set(a.lower().split()), set(b.lower().split())
        return len(a & b) / max(1, len(a))
    for name in [c for c in conditions if c.startswith("dpbart")]:
        own = np.mean([overlap(conditions[name][d], " ".join(docs[d]["original"].split()[:40]))
                       for d in range(len(docs))])
        other = np.mean([overlap(conditions[name][d], " ".join(docs[(d + 7) % len(docs)]["original"].split()[:40]))
                         for d in range(len(docs))])
        print(f"{name}: word overlap with own opening {own:.3f}, with another doc's {other:.3f}")


if __name__ == "__main__":
    main()
