# LoSanR: Low-dimensional Sanitization via Retrieval

Code accompanying the paper **"LoSanR: Free Lunches for Differentially Private Text Sanitization"**.

LoSanR sanitizes private text documents under local differential privacy (LDP) by:

1. Encoding sentences with a public sentence-transformer (`all-MiniLM-L6-v2`).
2. Projecting embeddings to a low-dimensional **PCA** space fitted on public data (zero privacy cost).
3. Adding calibrated Gaussian noise via the **Analytic Gaussian Mechanism** (Balle & Wang, 2018).
4. Retrieving the **nearest public-corpus sentence** as a replacement, with context-aware scoring across surrounding sentences.

The privacy budget is document-level: a document of L sentences spends ε/L per sentence under basic composition (`--composition_mode honest_doc`, the default everywhere).

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

Datasets (IMDB, Amazon Polarity, AG News) are downloaded through the `datasets` library into the Hugging Face cache (`$HF_HOME`). Every script is run from the repository root.

## Step 1: sentence cache and a small run

```bash
python run_experiment.py --dataset imdb --epsilons 100 250 500 1000 --pca_dim 16 --max_private 200
```

The first run splits every training document into sentences and embeds them once, into `results/imdb_all-MiniLM-L6-v2_cache/`. All other scripts read this cache. The run itself sanitizes 200 private documents and evaluates them with a logistic-regression classifier on sentence embeddings (the low-resource protocol of the paper's appendix). Results go to `results/imdb_pubAll_priv200_s42_pca16_honest_results.json`.

Other datasets: `--dataset amazon` (use `--delta 1e-7`) and `--dataset agnews`.

## Step 2: the main tables (BERT protocol)

The paper's main numbers use a common protocol with DP-BART: sanitize the full private half of the train split (12,500 documents), fine-tune `bert-base-uncased` on the sanitized text (max length 256, batch 32, 3 epochs, learning rate 2e-5), and report the final-epoch macro F1 on the unsanitized test split, averaged over seeds 0, 1 and 2. A seed sets both the DP noise and the fine-tuning.

```bash
# sanitize the 12,500 private IMDB documents with PCA-16 and context-aware retrieval
python protocol/sanitize_variant.py --dataset imdb --pca_dim 16 --retrieval ctx \
    --delta 1e-5 --noise_seed 0 --tag imdb_k16_ctx --epsilons 25 50 100 250 500 1000

# fine-tune BERT on one of the files (GPU)
python protocol/bert_finetune_full.py \
    --train_jsonl results/bert_eval/variants/imdb_k16_ctx_eps500_s0_sanitized.jsonl \
    --imdb_test_parquet $HF_HOME/hub/datasets--imdb/snapshots/*/plain_text/test-00000-of-00001.parquet \
    --out_json results/bert_eval/variants/finetunes/bert_imdb_k16_ctx_eps500_s0.json \
    --max_len 256 --batch_size 32 --epochs 3 --seed 0

# table rows from all finished fine-tunes
python protocol/make_bert_tables.py
```

`protocol/slurm/submit_bert_variants.sh` submits the whole grid on a Slurm cluster (PCA sweep, no-PCA rows, plain nearest-neighbor retrieval, public-corpus tiers, SanText, DP-BART, three datasets). Set `LOSANR_ROOT`, and `SBATCH_ACCOUNT` and `GRES` for your cluster.

Variants: `--pca_dim 0` disables PCA; `--retrieval nn` is plain nearest-neighbor retrieval; `--pool_tier rotten_tomatoes --pool_size 10000` swaps the public corpus (build the Rotten Tomatoes pool once with `python analysis/corpus_tiers.py --tiers rotten_tomatoes --sizes 0 --epsilons 100 --noise_seeds 0`).

## Baselines

**SanText** (Yue et al., 2021), our implementation with per-word metric-DP budgets over GloVe 6B 300d distances:

```bash
python protocol/santext_full.py --noise_seed 0 --epsilons 2 2.25 2.5 2.75 3    # 12,500 docs, for BERT
python attacks/santext_baseline.py --epsilons 2.25 2.5 3                       # 200 docs, for the attacks
```

GloVe is loaded from the sentence-transformers model `average_word_embeddings_glove.6B.300d` in the Hugging Face cache (download it once with `SentenceTransformer("average_word_embeddings_glove.6B.300d")`).

**DP-BART** (Igamberdiev and Habernal, 2023) is run with the authors' code at `github.com/trusthlt/dp-bart-private-rewriting` (DP-BART-PR+ checkpoints). Its rewritten IMDB train split is converted to our training format with `python protocol/dpbart_to_jsonl.py EPS rewritten_train.csv out.jsonl` and then fine-tuned like any other file.

## Empirical privacy

```bash
# model-free re-identification and the sanitized texts of the 200 private documents
python attacks/linkage_ablation.py --cache_dir results/imdb_all-MiniLM-L6-v2_cache \
    --out_dir results/linkage --epsilons 25 50 100 250 500 1000

# targets, decoys and conditions for the LLM attacks (DP-BART and SanText are optional extras)
python attacks/build_attack_inputs.py --extra santext_eps2.5=results/santext/santext_eps2.5_s0.jsonl
python attacks/matched_linkage.py            # decoys with the original's sentence count

# LLM attacks through OpenRouter (OPENROUTER_API_KEY in .env; the paper uses Claude Haiku 4.5)
python attacks/llm_attack.py --task reconstruct --conditions losanr_eps500 --model anthropic/claude-haiku-4.5
python attacks/llm_attack.py --task link --decoys lookalike_len --inputs results/attacks/inputs_matched.json \
    --conditions losanr_eps500 --model anthropic/claude-haiku-4.5

# scoring: does the reconstruction contain a rare name or number of the original?
python attacks/score_specifics.py --max_df 0.01
python attacks/entity_survival.py            # how many specifics survive in the sanitized text itself
python attacks/output_diversity.py           # share of unique replacement sentences
```

## Analyses

```bash
python analysis/retrieval_coherence.py --dataset imdb --delta 1e-5 --epsilons 25 50 100 250 500 1000
python analysis/retrieval_ablation.py        # retrieval ablation with GPT-2 perplexity (200 docs)
python analysis/corpus_tiers.py              # public-corpus mismatch grid (200 docs, logistic regression)
python analysis/figure_privacy_utility.py    # the privacy-versus-utility figure
```

## Key flags of `run_experiment.py`

| Flag | Default | Description |
|---|---|---|
| `--dataset` | `imdb` | One of `imdb`, `amazon`, `agnews`, `atis`, `snips`, `drugscom` |
| `--epsilons` | `100 250 500 1000` | Document-level ε budgets to sweep |
| `--delta` | `1e-5` | δ for the Analytic Gaussian Mechanism (the paper uses `1e-7` for Amazon) |
| `--pca_dim` | `16` | PCA target dimension (`0` disables PCA) |
| `--composition_mode` | `honest_doc` | `honest_doc` splits ε across sentences (paper default); `per_sentence` treats ε as per-sentence (the inflated reporting discussed in the paper's budget-accounting appendix) |
| `--max_private` | `200` | Number of private documents to sanitize |
| `--n_clusters` | `50` | k for the public-pool k-Means index |
| `--top_k_clusters` | `3` | Number of nearest clusters to search per query |

## Repository structure

```
losanr/
├── run_experiment.py          # sentence cache, 200-document runs, logistic-regression evaluation
├── src/
│   ├── common/
│   │   ├── embeddings.py      # SentenceEmbedder, PCAProjector
│   │   ├── clustering.py      # SentenceClusterIndex (k-Means over the public pool)
│   │   ├── dp_mechanisms.py   # Analytic Gaussian Mechanism
│   │   ├── data.py            # dataset loading, sentence splitter, text-only public corpora
│   │   └── evaluation.py      # cosine, BLEU, ROUGE, logistic-regression F1
│   └── losanr/
│       └── pipeline.py        # LoSanR pipeline (encode, project, noise, retrieve)
├── protocol/                  # BERT protocol: sanitize 12,500 docs, fine-tune, tables; SanText; DP-BART conversion
│   └── slurm/                 # job scripts for the full grid
├── attacks/                   # re-identification, LLM reconstruction and candidate selection, scoring
├── analysis/                  # coherence, retrieval ablation, public-corpus tiers, figure
├── requirements.txt
├── LICENSE                    # MIT
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
