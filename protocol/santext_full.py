"""SanText (Yue et al., 2021) on the full private half of IMDB, for BERT fine-tuning.

Same mechanism and vocabulary rule as attacks/santext_baseline.py (the 200-document run used
for the attacks): each word x is replaced by y sampled with probability proportional to
exp(-eps * d(x, y) / 2) over GloVe 6B 300d Euclidean distance; punctuation is kept; the
vocabulary is the --vocab_size most frequent public words plus the words of the documents
being sanitized; words without a GloVe vector get a random one.

GloVe comes from the sentence-transformers model average_word_embeddings_glove.6B.300d in the
Hugging Face cache ($HF_HOME). Requires the IMDB sentence cache written by run_experiment.py.

Output: {out_dir}/imdb_santext_eps{E}_s{noise_seed}_sanitized.jsonl with {"label", "text"}.
"""
import argparse
import glob
import json
import os
import pickle
import re
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.common.data import load_dataset_splits  # noqa: E402

TOKEN = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z]+)?|[^\sA-Za-z0-9]")
MODEL = "all-MiniLM-L6-v2"


def log(m):
    print(m, flush=True)


def load_glove(hf_home):
    base = glob.glob(os.path.join(
        hf_home, "hub", "models--sentence-transformers--average_word_embeddings_glove.6B.300d",
        "snapshots", "*", "0_WordEmbeddings"))[0]
    vocab = json.load(open(os.path.join(base, "whitespacetokenizer_config.json"), encoding="utf-8"))["vocab"]
    from safetensors.torch import load_file
    W = next(iter(load_file(os.path.join(base, "model.safetensors")).values())).numpy().astype(np.float32)
    log(f"GloVe: {len(vocab)} words, matrix {W.shape}")
    return {w: i for i, w in enumerate(vocab)}, W


def is_word(tok):
    return tok[0].isalnum()


def detok(tokens):
    return re.sub(r"\s+([.,!?;:%)\]])", r"\1", " ".join(tokens))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epsilons", type=float, nargs="+", default=[2, 2.25, 2.5, 2.75, 3])
    ap.add_argument("--noise_seed", type=int, required=True)
    ap.add_argument("--vocab_size", type=int, default=20000)
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--out_dir", default="results/bert_eval/variants")
    ap.add_argument("--chunk", type=int, default=1000)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    train_all = load_dataset_splits("imdb")["train"]
    labels = [l for _, l in train_all]
    with open(os.path.join(args.results_dir, f"imdb_{MODEL}_cache", "train_doc_sentences.pkl"), "rb") as f:
        doc_sents = pickle.load(f)
    perm = np.random.RandomState(42).permutation(len(doc_sents))
    half = int(len(doc_sents) * 0.5)
    public, private = perm[:half], perm[half:]

    freq = Counter()
    for idx in public:
        for s in doc_sents[idx]:
            freq.update(t.lower() for t in TOKEN.findall(s) if is_word(t))
    priv_tok = [[TOKEN.findall(s) for s in doc_sents[idx]] for idx in private]
    counts = Counter(t.lower() for d in priv_tok for s in d for t in s if is_word(t))
    vocab = [w for w, _ in freq.most_common(args.vocab_size)]
    seen = set(vocab)
    vocab += sorted(w for w in counts if w not in seen)
    v_index = {w: i for i, w in enumerate(vocab)}
    types = sorted(counts)
    log(f"private docs {len(private)}, word tokens {sum(counts.values())}, types {len(types)}, vocab {len(vocab)}")

    g_index, G = load_glove(os.environ["HF_HOME"])
    rng0 = np.random.RandomState(0)
    scale = float(G.std())
    E = np.zeros((len(vocab), G.shape[1]), dtype=np.float32)
    oov = 0
    for i, w in enumerate(vocab):
        j = g_index.get(w)
        if j is None:
            E[i] = rng0.normal(0, scale, size=G.shape[1])
            oov += 1
        else:
            E[i] = G[j]
    log(f"words without a GloVe vector: {oov} ({oov / len(vocab):.1%})")
    sq = (E ** 2).sum(axis=1)

    for eps in args.epsilons:
        rng = np.random.RandomState(args.noise_seed)
        repl = {}
        for a in range(0, len(types), args.chunk):
            chunk = types[a:a + args.chunk]
            r = np.array([v_index[w] for w in chunk])
            d2 = sq[r][:, None] + sq[None, :] - 2.0 * (E[r] @ E.T)
            logit = -0.5 * eps * np.sqrt(np.maximum(d2, 0.0)).astype(np.float64)
            logit -= logit.max(axis=1, keepdims=True)
            P = np.exp(logit)
            P /= P.sum(axis=1, keepdims=True)
            for w, p in zip(chunk, P):
                repl[w] = list(rng.choice(len(vocab), size=counts[w], p=p))
        out = os.path.join(args.out_dir, f"imdb_santext_eps{eps:g}_s{args.noise_seed}_sanitized.jsonl")
        same = total = 0
        with open(out + ".tmp", "w", encoding="utf-8") as f:
            for idx, d in zip(private, priv_tok):
                sents = []
                for s in d:
                    toks = []
                    for t in s:
                        if is_word(t):
                            y = vocab[repl[t.lower()].pop()]
                            same += int(y == t.lower())
                            total += 1
                            toks.append(y)
                        else:
                            toks.append(t)
                    sents.append(detok(toks))
                f.write(json.dumps({"label": int(labels[idx]), "text": " ".join(sents)}) + "\n")
        os.replace(out + ".tmp", out)
        log(f"eps={eps:g}: wrote {out}; words kept unchanged {same / max(1, total):.1%}")
    log("ALL_SANITIZE_DONE")


if __name__ == "__main__":
    sys.exit(main())
