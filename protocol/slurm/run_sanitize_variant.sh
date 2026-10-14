#!/bin/bash
#SBATCH --job-name=san-variant
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# One sanitization job (CPU) for the BERT-protocol grid. Submitted by submit_bert_variants.sh.
# Required env: LOSANR_ROOT (repository root), MODE, NSEED.
# MODE=losanr  : env TAG, DATASET, K, RETRIEVAL, DELTA; optional POOL_TIER, POOL_SIZE, EPSILONS
# MODE=santext : optional EPSILONS
# $LOSANR_ROOT/cluster_env.sh is sourced if present (activate the venv, set HF_HOME there).
set -euo pipefail
MODE="${MODE:?need MODE}"
NSEED="${NSEED:?need NSEED}"
cd "${LOSANR_ROOT:?set LOSANR_ROOT to the repository root}"
mkdir -p logs results/bert_eval/variants
[ -f cluster_env.sh ] && source cluster_env.sh

if [ "$MODE" = "losanr" ]; then
    EXTRA=""
    [ -n "${POOL_TIER:-}" ] && EXTRA="--pool_tier $POOL_TIER --pool_size ${POOL_SIZE:-0}"
    python protocol/sanitize_variant.py --dataset "$DATASET" --pca_dim "$K" --retrieval "$RETRIEVAL" \
        --delta "$DELTA" --noise_seed "$NSEED" --tag "$TAG" \
        --epsilons ${EPSILONS:-100 250 500 1000} $EXTRA
else
    python protocol/santext_full.py --noise_seed "$NSEED" --epsilons ${EPSILONS:-2 2.25 2.5 2.75 3}
fi
