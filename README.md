# LoSanR: Low-dimensional Sanitization via Retrieval

Code accompanying the paper **"LoSanR: Free Lunches for Differentially Private Text Sanitization"**.

LoSanR sanitizes private text documents under local differential privacy (LDP) by:

1. Encoding sentences with a public sentence-transformer (`all-MiniLM-L6-v2`).
2. Projecting embeddings to a low-dimensional **PCA** space fitted on public data (zero privacy cost).
3. Adding calibrated Gaussian noise via the **Analytic Gaussian Mechanism** (Balle & Wang, 2018).
4. Retrieving the **nearest public-corpus sentence** as a replacement, with context-aware scoring across surrounding sentences.

## Install

```bash
git clone https://github.com/thomouvic/losanr.git
cd losanr
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

First-run NLTK setup (one-time):
```python
import nltk; nltk.download("punkt_tab")
```

## Quickstart: reproduce the paper's IMDB results

```bash
python run_experiment.py \
    --dataset imdb \
    --epsilons 100 250 500 1000 \
    --pca_dim 16 \
    --composition_mode honest_doc \
    --max_private 200
```

This runs LoSanR (PCA-16, context-aware retrieval, document-level composition accounting) on 200 IMDB documents across the four ε budgets from the paper. Results are written to `results/imdb_pubAll_priv200_s42_pca16_honest_results.json`.

Run for Amazon Polarity or AG News:

```bash
python run_experiment.py --dataset amazon  --epsilons 100 250 500 1000 --pca_dim 16 --composition_mode honest_doc --max_private 200
python run_experiment.py --dataset agnews  --epsilons 100 250 500 1000 --pca_dim 16 --composition_mode honest_doc --max_private 200
```

## Key flags

| Flag | Default | Description |
|---|---|---|
| `--dataset` | `imdb` | One of `imdb`, `amazon`, `agnews`, `atis`, `snips`, `drugscom` |
| `--epsilons` | `100 250 500 1000` | Document-level ε budgets to sweep |
| `--pca_dim` | `16` | PCA target dimension (`0` disables PCA, useful for ablation) |
| `--composition_mode` | `honest_doc` | `honest_doc` splits ε across sentences (paper default); `per_sentence` treats ε as per-sentence (reproduces the inflated-budget reporting we critique in §2.5) |
| `--max_private` | `200` | Number of private documents to sanitize |
| `--n_clusters` | `50` | k for the public-pool k-Means index |
| `--top_k_clusters` | `3` | Number of nearest clusters to search per query |
| `--delta` | `1e-5` | δ for the AGM (per the paper's per-dataset choice) |

## Repository structure

```
losanr/
├── run_experiment.py       # Main experiment runner (sweeps ε, evaluates utility)
├── src/
│   ├── common/
│   │   ├── embeddings.py      # SentenceEmbedder, PCAProjector
│   │   ├── clustering.py      # SentenceClusterIndex (k-Means over public pool)
│   │   ├── dp_mechanisms.py   # Analytic Gaussian Mechanism, Exponential Mechanism
│   │   ├── data.py            # Dataset loading + regex sentence splitter
│   │   └── evaluation.py      # Cosine sim, BLEU, ROUGE, downstream-F1 evaluation
│   └── losanr/
│       └── pipeline.py        # LoSanR pipeline (encode → project → noise → retrieve)
├── requirements.txt
├── LICENSE                 # MIT
└── README.md
```

## Citation

```bibtex
@misc{losanr2026,
  title  = {{LoSanR}: Free Lunches for Differentially Private Text Sanitization},
  author = {TODO},
  year   = {2026},
  note   = {Under submission}
}
```

*BibTeX entry will be filled in upon publication.*
