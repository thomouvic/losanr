"""Output diversity of LoSanR on the 200 IMDB private documents (full public pool).

For each budget: share of replacement sentences that are unique across the 200 sanitized
documents, how often the most-reused sentence appears, and whether any two sanitized
documents are identical. Reads results/linkage/texts/losanr_pca16_ctx_eps{E}_s0.jsonl.
"""
import json
import os
from collections import Counter

print(f"{'eps':>5s} {'sents':>6s} {'unique':>7s} {'max reuse':>9s} {'identical docs':>14s}")
for eps in (10, 25, 50, 100, 250, 500, 1000):
    p = f"results/linkage/texts/losanr_pca16_ctx_eps{eps}_s0.jsonl"
    if not os.path.exists(p):
        continue
    rows = [json.loads(l) for l in open(p, encoding="utf-8")]
    sents = [s for r in rows for s in r["sanitized_sentences"]]
    c = Counter(sents)
    unique = sum(1 for s, k in c.items() if k == 1) / len(sents)
    docs = Counter(" ".join(r["sanitized_sentences"]) for r in rows)
    dup = sum(k - 1 for k in docs.values() if k > 1)
    print(f"{eps:5d} {len(sents):6d} {unique:7.1%} {max(c.values()):9d} {dup:14d}")
