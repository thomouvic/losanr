"""SanText word-level baseline (Yue et al., 2021) on the 200 IMDB private documents.

Each word x is replaced by a word y sampled from the vocabulary with probability
proportional to exp(-eps * d(x, y) / 2), where d is the Euclidean distance between GloVe
embeddings (metric local DP per token). Punctuation is kept. Sentences are processed one
by one from the cached sentence split, so sentence alignment with the original is kept.

Vocabulary: the --vocab_size most frequent words of the public half plus every word of the
private documents. Words missing from GloVe get a random vector, as in the reference code.

Outputs in --out_dir:
  santext_eps{E}_s0.jsonl          sanitized texts (doc_idx, src_idx, sanitized_sentences, n_words)
  santext_eps{E}_s0_docemb.npy     mean-of-sentence MiniLM embedding of the first --max_sents
                                   sanitized sentences (for the embedding linkage attacker)
  santext_summary.json             logistic-regression F1 per eps over noise seeds, word counts

GloVe comes from the sentence-transformers model average_word_embeddings_glove.6B.300d in the
Hugging Face cache ($HF_HOME). Run from the repository root.
"""
import argparse
import json
import os
import re
import sys
from collections import Counter
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.common.embeddings import SentenceEmbedder  # noqa: E402
from src.common.evaluation import downstream_f1  # noqa: E402
from analysis.corpus_tiers import load_private_setup, build_dataset_pool  # noqa: E402

TOKEN = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z]+)?|[^\sA-Za-z0-9]")


def log(m):
    print(m, flush=True)


def load_glove(hf_home):
    import glob
    import torch
    base = glob.glob(os.path.join(
        hf_home, "hub", "models--sentence-transformers--average_word_embeddings_glove.6B.300d",
        "snapshots", "*", "0_WordEmbeddings"))[0]
    vocab = json.load(open(os.path.join(base, "whitespacetokenizer_config.json"), encoding="utf-8"))["vocab"]
    st = os.path.join(base, "model.safetensors")
    if os.path.exists(st):
        from safetensors.torch import load_file
        sd = load_file(st)
    else:
        sd = torch.load(os.path.join(base, "pytorch_model.bin"), map_location="cpu")
    W = next(iter(sd.values())).numpy().astype(np.float32)
    log(f"GloVe: {len(vocab)} words, matrix {W.shape}")
    return {w: i for i, w in enumerate(vocab)}, W


def is_word(tok):
    return tok[0].isalnum()


def detok(tokens):
    s = " ".join(tokens)
    return re.sub(r"\s+([.,!?;:%)\]])", r"\1", s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epsilons", type=float, nargs="+", default=[1, 2, 3])
    ap.add_argument("--noise_seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--vocab_size", type=int, default=20000)
    ap.add_argument("--max_sents", type=int, default=15)
    ap.add_argument("--out_dir", default="results/santext")
    ap.add_argument("--output_dir", default="results")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    setup = SimpleNamespace(private_dataset="imdb", output_dir=args.output_dir,
                            model="all-MiniLM-L6-v2", seed=42, public_ratio=0.5, max_private=200)
    embedder = SentenceEmbedder(setup.model)
    priv = load_private_setup(setup, embedder)
    pool_sentences, _ = build_dataset_pool("imdb", embedder, args.output_dir, setup.model, 42, 0.5)

    # private tokens, per document and sentence
    priv_tok = [[TOKEN.findall(s) for s in sents] for sents in priv["doc_sentences"]]
    priv_types = sorted({t.lower() for d in priv_tok for s in d for t in s if is_word(t)})

    freq = Counter()
    for s in pool_sentences:
        freq.update(t.lower() for t in TOKEN.findall(s) if is_word(t))
    vocab = [w for w, _ in freq.most_common(args.vocab_size)]
    seen = set(vocab)
    vocab += [w for w in priv_types if w not in seen]
    v_index = {w: i for i, w in enumerate(vocab)}
    log(f"vocab {len(vocab)} (public top {args.vocab_size} + {len(vocab) - args.vocab_size} private-only); "
        f"private word types {len(priv_types)}")

    g_index, G = load_glove(os.environ["HF_HOME"])
    rng0 = np.random.RandomState(0)
    E = np.zeros((len(vocab), G.shape[1]), dtype=np.float32)
    oov = 0
    scale = float(G.std())
    for i, w in enumerate(vocab):
        j = g_index.get(w)
        if j is None:
            E[i] = rng0.normal(0, scale, size=G.shape[1])
            oov += 1
        else:
            E[i] = G[j]
    log(f"words without a GloVe vector: {oov} ({oov / len(vocab):.1%})")

    # distances from every private word type to the whole vocabulary
    rows = np.array([v_index[w] for w in priv_types])
    sq = (E ** 2).sum(axis=1)
    D = np.empty((len(rows), len(vocab)), dtype=np.float32)
    for a in range(0, len(rows), 1000):
        r = rows[a:a + 1000]
        d2 = sq[r][:, None] + sq[None, :] - 2.0 * (E[r] @ E.T)
        D[a:a + 1000] = np.sqrt(np.maximum(d2, 0.0))
    type_row = {w: i for i, w in enumerate(priv_types)}
    log(f"distance matrix {D.shape}; median distance {float(np.median(D[:200])):.2f}, "
        f"max {float(D.max()):.2f}")

    counts = Counter(t.lower() for d in priv_tok for s in d for t in s if is_word(t))
    n_words = [sum(1 for s in d for t in s if is_word(t)) for d in priv_tok]
    summary = {"meta": vars(args), "mean_words_per_doc": float(np.mean(n_words)),
               "max_glove_distance": float(D.max()), "eps": {}}

    for eps in args.epsilons:
        f1s, kept = [], []
        for seed in args.noise_seeds:
            rng = np.random.RandomState(seed)
            # sample all replacements for each word type at once
            repl = {}
            for w, c in counts.items():
                logit = -0.5 * eps * D[type_row[w]].astype(np.float64)
                p = np.exp(logit - logit.max())
                p /= p.sum()
                repl[w] = list(rng.choice(len(vocab), size=c, p=p))
            same = total = 0
            out_docs = []
            for d in priv_tok:
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
                out_docs.append(sents)
            kept.append(same / max(1, total))
            ds = downstream_f1(train_texts=[" ".join(s) for s in out_docs], train_labels=priv["labels"],
                               test_texts=priv["test_texts"], test_labels=priv["test_labels"],
                               embedder=embedder, test_embeddings=priv["test_emb_norm"])
            f1s.append(ds["f1_macro"])
            log(f"eps={eps} seed={seed}: F1={ds['f1_macro']:.3f} words kept unchanged={kept[-1]:.1%}")
            if seed == args.noise_seeds[0]:
                tag = f"santext_eps{eps:g}_s{seed}"
                with open(os.path.join(args.out_dir, tag + ".jsonl"), "w", encoding="utf-8") as f:
                    for di, sents in enumerate(out_docs):
                        f.write(json.dumps({"doc_idx": di, "sanitized_sentences": sents,
                                            "sanitized_text": " ".join(sents[:args.max_sents]),
                                            "n_words": n_words[di]}) + "\n")
                embs = np.zeros((len(out_docs), embedder.dim), dtype=np.float32)
                for di, sents in enumerate(out_docs):
                    if sents:
                        m = embedder.embed(sents[:args.max_sents], normalize=True).mean(axis=0)
                        embs[di] = m / max(np.linalg.norm(m), 1e-10)
                np.save(os.path.join(args.out_dir, tag + "_docemb.npy"), embs)
        summary["eps"][f"{eps:g}"] = {"f1": f1s, "f1_mean": float(np.mean(f1s)), "f1_std": float(np.std(f1s)),
                                      "kept_unchanged": float(np.mean(kept)),
                                      "eps_doc_composed_mean": float(eps * np.mean(n_words))}
        with open(os.path.join(args.out_dir, "santext_summary.json"), "w") as f:
            json.dump(summary, f, indent=1)
    log(json.dumps(summary["eps"], indent=1))
    log("Done.")


if __name__ == "__main__":
    main()
