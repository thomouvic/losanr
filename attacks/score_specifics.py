"""Specifics of the original (rare names and numbers) in the sanitized text and in the
attacker's reconstruction. Replaces the scoring of score_reconstruct.py, whose
case-sensitive matching missed names that SanText keeps in lowercase.

A specific is a token of the original that is either a number or a capitalized word not at
the start of its sentence, and that is rare in the public pool (appears in at most
--max_df of the public documents). Matching against sanitized text and reconstructions is
case-insensitive.

Per condition (IMDB, 200 private documents):
  in text    share of documents whose sanitized text contains at least one specific of the
             original, and (second number) the mean share of specifics it contains
  chance     the same, measured against another document's sanitized text
  attacker   share of documents whose reconstruction contains at least one specific of the
             original (copied or inferred), with its guessing baseline
  inferred   share of documents whose reconstruction contains a specific that the sanitized
             text does not contain, with its guessing baseline

Usage (from the repository root): python attacks/score_specifics.py [--max_df 0.01]
Reconstructions are taken from responses whose model name contains --model_substring.
"""
import argparse
import json
import os
import pickle
import re
from collections import Counter, defaultdict

import numpy as np

STOP = set("""a an the and or but if while of at by for with about against between into through during
before after above below to from up down in out on off over under again further then once here there
when where why how all any both each few more most other some such no nor not only own same so than
too very s t can will just don should now i me my myself we our ours ourselves you your yours yourself
he him his himself she her hers herself it its itself they them their theirs themselves what which who
whom this that these those am is are was were be been being have has had having do does did doing
would could should it's i'm don't didn't can't isn't wasn't one also much many even really like get got
movie film movies films""".split())
TOKEN = re.compile(r"[A-Za-z][A-Za-z'\-]+|\d[\d,.]*")


def candidates(text):
    """Numbers and capitalized non-initial words of text, lowercased."""
    out = set()
    for sent in re.split(r"(?<=[.!?])\s+", text):
        toks = TOKEN.findall(sent)
        for i, t in enumerate(toks):
            if t[0].isdigit():
                t = t.strip(",.")
                if t:
                    out.add(t)
            elif i > 0 and t[0].isupper() and t.lower() not in STOP and len(t) > 2:
                out.add(t.lower())
    return out


def words(text):
    return {t.strip(",.").lower() for t in TOKEN.findall(text)}


def public_df(cache_dir, seed=42):
    """Document frequency of lowercased tokens over the public half of the train split."""
    with open(os.path.join(cache_dir, "train_doc_sentences.pkl"), "rb") as f:
        all_doc_sentences = pickle.load(f)
    n = len(all_doc_sentences)
    rng = np.random.RandomState(seed)
    public = rng.permutation(n)[: n // 2]
    df = Counter()
    for idx in public:
        df.update(words(" ".join(all_doc_sentences[idx])))
    return df, len(public)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_df", type=float, default=0.01, help="max share of public docs a specific may appear in")
    ap.add_argument("--inputs", default="results/attacks/inputs.json")
    ap.add_argument("--responses", default="results/attacks/responses.jsonl")
    ap.add_argument("--cache_dir", default="results/imdb_all-MiniLM-L6-v2_cache")
    ap.add_argument("--out", default="results/attacks/specifics_scores.json")
    ap.add_argument("--model_substring", default="haiku", help="score responses of models whose name contains this")
    args = ap.parse_args()

    data = json.load(open(args.inputs, encoding="utf-8"))
    docs = data["docs"]
    n = len(docs)
    df, n_public = public_df(args.cache_dir)
    limit = args.max_df * n_public

    spec = []
    for d in docs:
        spec.append({s for s in candidates(d["original"]) if df.get(s, 0) <= limit})
    n_with = sum(1 for s in spec if s)
    print(f"public docs {n_public}, max_df {args.max_df:g} (<= {limit:.0f} docs); "
          f"documents with at least one specific: {n_with}/{n}; "
          f"mean specifics per document {np.mean([len(s) for s in spec]):.1f}")

    def hit(s, text):
        w = words(text)
        return {x for x in s if x in w}

    out = {"max_df": args.max_df, "n_docs_with_specifics": n_with, "conditions": {}}
    print(f"\n{'condition':22s} | {'in text':>14s} | {'chance':>14s}")
    for cond in sorted(data["conditions"]):
        texts = data["conditions"][cond]
        a, b = [], []
        for di in range(n):
            if not spec[di]:
                continue
            h = hit(spec[di], texts[di])
            h0 = hit(spec[di], texts[(di + 7) % n])
            a.append((int(bool(h)), len(h) / len(spec[di])))
            b.append((int(bool(h0)), len(h0) / len(spec[di])))
        m = lambda X, j: float(np.mean([x[j] for x in X]))
        out["conditions"][cond] = {"in_text_any": m(a, 0), "in_text_share": m(a, 1),
                                   "chance_any": m(b, 0), "chance_share": m(b, 1)}
        print(f"{cond:22s} | {m(a,0):6.2f} {m(a,1):7.3f} | {m(b,0):6.2f} {m(b,1):7.3f}")

    rows = defaultdict(list)
    for line in open(args.responses, encoding="utf-8"):
        r = json.loads(line)
        if r["task"] == "reconstruct" and "error" not in r and args.model_substring in r["model"]:
            rows[r["condition"]].append(r)
    print(f"\n{'condition':22s} {'n':>4s} | {'attacker':>8s} {'guess':>6s} | {'inferred':>8s} {'guess':>6s}")
    for cond, rs in sorted(rows.items()):
        A, G, I, GI = [], [], [], []
        for r in rs:
            di = r["doc_idx"]
            if not spec[di]:
                continue
            san = data["conditions"][cond][di]
            got = hit(spec[di], r["text"])
            A.append(int(bool(got)))
            I.append(int(bool(got - hit(spec[di], san))))
            dj = (di + 7) % n
            if spec[dj]:
                got0 = hit(spec[dj], r["text"])
                G.append(int(bool(got0)))
                GI.append(int(bool(got0 - hit(spec[dj], san))))
        res = {"n": len(rs), "attacker_any": float(np.mean(A)), "guess_any": float(np.mean(G)),
               "inferred_any": float(np.mean(I)), "guess_inferred": float(np.mean(GI))}
        out["conditions"][cond].update(res)
        print(f"{cond:22s} {len(rs):4d} | {res['attacker_any']:8.2f} {res['guess_any']:6.2f} | "
              f"{res['inferred_any']:8.2f} {res['guess_inferred']:6.2f}")
    json.dump(out, open(args.out, "w"), indent=1)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
