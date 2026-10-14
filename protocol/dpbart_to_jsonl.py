"""Turn DP-BART's rewritten IMDB training set into BERT training files for our protocol.

DP-BART PR+ (Igamberdiev and Habernal, 2023; github.com/trusthlt/dp-bart-private-rewriting)
rewrites the full IMDB train split in dataset order. This selects the rows of our 12,500
private documents (split seed 42, second half) and writes {"label", "text"} JSONL, the same
format as the LoSanR sanitized files.

Usage: python protocol/dpbart_to_jsonl.py EPS path/to/rewritten_train.csv out.jsonl
"""
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.common.data import load_dataset_splits  # noqa: E402

eps, src, out = sys.argv[1], sys.argv[2], sys.argv[3]
csv.field_size_limit(10 ** 9)
rows = list(csv.reader(open(src, encoding="utf-8", errors="ignore")))
train = load_dataset_splits("imdb")["train"]
assert len(rows) == len(train), f"{len(rows)} rewritten rows vs {len(train)} train docs"
mism = sum(1 for r, (_, l) in zip(rows, train) if int(r[0]) != int(l))
assert mism == 0, f"{mism} label mismatches: rewritten rows are not in train order"

perm = np.random.RandomState(42).permutation(len(train))
private = perm[int(len(train) * 0.5):]
empty = 0
with open(out, "w", encoding="utf-8") as f:
    for idx in private:
        text = " ".join(rows[idx][1].split()) if len(rows[idx]) > 1 else ""
        empty += int(not text)
        f.write(json.dumps({"label": int(train[idx][1]), "text": text}) + "\n")
print(f"eps={eps}: wrote {len(private)} rows to {out}; empty rewrites {empty} ({empty / len(private):.1%})")
