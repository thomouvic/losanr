#!/bin/bash
#SBATCH --job-name=bert-seed
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# One BERT fine-tune (GPU) of the common evaluation protocol: 12,500 sanitized documents,
# bert-base-uncased, max length 256, batch 32, 3 epochs, final-epoch macro F1.
# Required env: LOSANR_ROOT, DATASET (imdb|amazon|agnews), JSONL, OUT_JSON, SEED.
# Optional: NUM_LABELS (default 2; 4 for agnews). The test split is read from the Hugging
# Face cache ($HF_HOME, set in cluster_env.sh), as downloaded by the datasets library.
set -euo pipefail
DATASET="${DATASET:?need DATASET}"
JSONL="${JSONL:?need JSONL}"
OUT_JSON="${OUT_JSON:?need OUT_JSON}"
SEED="${SEED:?need SEED}"
NUM_LABELS="${NUM_LABELS:-2}"

cd "${LOSANR_ROOT:?set LOSANR_ROOT to the repository root}"
mkdir -p logs "$(dirname "$OUT_JSON")"
[ -f cluster_env.sh ] && source cluster_env.sh
HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

case "$DATASET" in
    imdb)
        TEST_PARQUET=$(ls "$HF_HOME"/hub/datasets--imdb/snapshots/*/plain_text/test-00000-of-00001.parquet | head -1)
        ;;
    amazon)
        TEST_PARQUET=$(ls "$HF_HOME"/hub/datasets--amazon_polarity/snapshots/*/amazon_polarity/test-00000-of-00001.parquet | head -1)
        ;;
    agnews)
        TEST_PARQUET=$(ls "$HF_HOME"/hub/datasets--ag_news/snapshots/*/data/test-00000-of-00001.parquet | head -1)
        ;;
    *)
        echo "Unknown dataset: $DATASET"; exit 1;;
esac

python -c "import torch; print('device:', torch.cuda.get_device_properties(0).name)"
echo "JSONL=$JSONL  OUT=$OUT_JSON  NUM_LABELS=$NUM_LABELS  SEED=$SEED  TEST=$TEST_PARQUET"
python protocol/bert_finetune_full.py \
    --train_jsonl "$JSONL" \
    --imdb_test_parquet "$TEST_PARQUET" \
    --out_json "$OUT_JSON" \
    --num_labels "$NUM_LABELS" \
    --seed "$SEED" \
    --max_len 256 \
    --batch_size 32 \
    --epochs 3
