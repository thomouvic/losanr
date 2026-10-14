"""Print the BERT-protocol table rows for the paper from results/bert_eval/variants/finetunes.

Each cell is the mean final-epoch macro F1 over the noise/training seeds available. Cells
with fewer than three seeds are flagged, and the largest standard deviation per table is
reported so captions can state it. File names follow protocol/slurm/submit_bert_variants.sh:
bert_{tag}_eps{E}_s{seed}.json.

Usage (from the repository root): python protocol/make_bert_tables.py
"""
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np

D = "results/bert_eval/variants/finetunes"
EPS = ["25", "50", "100", "250", "500", "1000"]

cells = defaultdict(dict)
for p in glob.glob(os.path.join(D, "bert_*.json")):
    m = re.match(r"bert_(.+)_eps([0-9.]+)_s(\d)\.json", os.path.basename(p))
    if m:
        cells[(m.group(1), m.group(2))][int(m.group(3))] = json.load(open(p))["final_epoch_macro_f1"]

seen_std = []
flags = []


def cell(tag, e):
    v = list(cells.get((tag, e), {}).values())
    if not v:
        return "---"
    if len(v) < 3:
        flags.append(f"{tag} eps={e}: {len(v)} seed(s)")
    else:
        seen_std.append(np.std(v))
    return f"{np.mean(v):.3f}"[1:]


def row(label, tag, eps=EPS):
    return f"{label} & " + " & ".join(cell(tag, e) for e in eps) + r" \\"


def table(title, rows):
    # rows were built (and seen_std filled) while evaluating the argument
    print(f"% ---- {title}")
    for r in rows:
        print(r)
    if seen_std:
        print(f"% std over 3-seed cells: median {np.median(seen_std):.3f}, largest {max(seen_std):.3f}")
    seen_std.clear()
    print()


table("PCA sweep, IMDB (context-aware retrieval)",
      [row(f"$k{{=}}{k}$", f"imdb_k{k}_ctx") for k in (8, 16, 32, 64)] + [row("No PCA ($d{=}384$)", "imdb_k0_ctx")])
table("PCA sweep, IMDB, small k (only run from eps=100)",
      [row(f"$k{{=}}{k}$", f"imdb_k{k}_ctx") for k in (2, 4)])
table("Cross-dataset",
      [row(f"{ds} PCA-16", f"{ds}_k16_ctx") for ds in ("imdb", "amazon", "agnews")]
      + [row(f"{ds} no PCA", f"{ds}_k0_ctx") for ds in ("imdb", "amazon", "agnews")])
table("Retrieval, IMDB PCA-16",
      [row("Plain NN", "imdb_k16_nn"), row("Context", "imdb_k16_ctx"),
       row("Plain NN, no PCA", "imdb_k0_nn"), row("Context, no PCA", "imdb_k0_ctx")])
table("Public-corpus tiers, IMDB private set, 10k-sentence pools, PCA-16",
      [row(lab, f"imdb_k16_ctx_pool-{t}10k") for t, lab in
       (("imdb", "IMDB (in-distribution)"), ("rotten_tomatoes", "Rotten Tomatoes (movie reviews)"),
        ("amazon", "Amazon (product reviews)"), ("agnews", "AG News (news)"))]
      + [row("IMDB, full pool (134k)", "imdb_k16_ctx")])
table("DP-BART rerun under our protocol (IMDB)", [row("DP-BART (rerun)", "imdb_dpbart")])
table("SanText (IMDB), per-word eps 2, 2.25, 2.5, 2.75, 3",
      [row("SanText", "imdb_santext", ["2", "2.25", "2.5", "2.75", "3"])])
table("Low budget, IMDB PCA-16", [row("eps=10", "imdb_k16_ctx", ["10"])])

print("% cells with fewer than three seeds:")
for f in sorted(set(flags)):
    print("%  ", f)

# no-DP reference: BERT trained on the same 12,500 documents in their original form
# (bert_{dataset}_original_s{seed}.json, produced by running bert_finetune_full.py on a
# {"label", "text"} file of the unsanitized private half)
for ds in ("imdb", "amazon", "agnews"):
    paths = [os.path.join(D, f"bert_{ds}_original_s{s}.json") for s in (0, 1, 2)]
    if all(os.path.exists(p) for p in paths):
        v = [json.load(open(p))["final_epoch_macro_f1"] for p in paths]
        print(f"% no-DP reference {ds}: {np.mean(v):.3f} +- {np.std(v):.3f}")
