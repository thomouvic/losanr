#!/bin/bash
# BERT-protocol grid behind the paper's tables, 3 seeds per cell. Run from the repository
# root on a Slurm cluster:  LOSANR_ROOT=$PWD bash protocol/slurm/submit_bert_variants.sh
#   A. PCA sweep on IMDB: k in {2,4,8,16,32,64} and no PCA
#   B. no-PCA rows for Amazon and AG News
#   C. plain nearest-neighbor retrieval on IMDB, with PCA-16 and without PCA
#   D. public-corpus tiers at 10k sentences (IMDB, Rotten Tomatoes, Amazon, AG News)
#   E. SanText word-level baseline on IMDB at five per-word budgets
#   F. DP-BART PR+ rewrites of IMDB under our protocol (JSONL from protocol/dpbart_to_jsonl.py)
#   G. PCA-16 context-aware rows for IMDB, Amazon and AG News (the main rows)
# Each sanitization job (CPU) covers all budgets for one (variant, seed); the BERT jobs
# depend on it. Cells whose output JSON exists are skipped.
# Env: LOSANR_ROOT (required); SECTIONS (default "A B C D E F G"); EPS_LIST (default
# "100 250 500 1000"; the paper also runs 25 and 50); SBATCH_ACCOUNT and GRES for your
# cluster (e.g. GRES=gpu:a100:1); SUFFIX to tag job names.
# Before section D, build the Rotten Tomatoes pool cache once:
#   python analysis/corpus_tiers.py --tiers rotten_tomatoes --sizes 0 --epsilons 100 --noise_seeds 0
set -euo pipefail
cd "${LOSANR_ROOT:?set LOSANR_ROOT to the repository root}"
export LOSANR_ROOT
V=results/bert_eval/variants
mkdir -p logs "$V/finetunes"
GRES="${GRES:-gpu:1}"
ACCT=""
[ -n "${SBATCH_ACCOUNT:-}" ] && ACCT="--account=$SBATCH_ACCOUNT"
SECTIONS="${SECTIONS:-A B C D E F G}"
EPS_LIST="${EPS_LIST:-100 250 500 1000}"
SUFFIX="${SUFFIX:-}"
N0=$(cat variant_jobs.txt 2>/dev/null | wc -l)

want() {
    case " $SECTIONS " in *" $1 "*) return 0 ;; *) return 1 ;; esac
}

bert() {  # DEP NAME DATASET JSONL OUT NUM_LABELS SEED
    local DEP=$1 NAME=$2 DS=$3 JSONL=$4 OUT=$5 NL=$6 S=$7 D=""
    [ -f "$OUT" ] && return 0
    [ -n "$DEP" ] && D="--dependency=afterok:$DEP"
    local J
    J=$(sbatch --parsable $ACCT $D --gres=$GRES --job-name="$NAME" \
        --export=ALL,DATASET=$DS,JSONL=$JSONL,OUT_JSON=$OUT,NUM_LABELS=$NL,SEED=$S \
        protocol/slurm/run_bert_seed.sh)
    echo "$NAME $J" >> variant_jobs.txt
}

losanr_one() {  # TAG DATASET K RETRIEVAL DELTA NUM_LABELS SEED [POOL_TIER POOL_SIZE]
    local TAG=$1 DS=$2 K=$3 R=$4 DELTA=$5 NL=$6 S=$7 PT=${8:-} PS=${9:-0} E SAN
    SAN=$(sbatch --parsable $ACCT --job-name="san-$TAG$SUFFIX-s$S" \
        --export="ALL,MODE=losanr,TAG=$TAG,DATASET=$DS,K=$K,RETRIEVAL=$R,DELTA=$DELTA,NSEED=$S,POOL_TIER=$PT,POOL_SIZE=$PS,EPSILONS=$EPS_LIST" \
        protocol/slurm/run_sanitize_variant.sh)
    echo "san-$TAG$SUFFIX-s$S $SAN" >> variant_jobs.txt
    for E in $EPS_LIST; do
        bert "$SAN" "bv-$TAG$SUFFIX-e$E-s$S" "$DS" "$V/${TAG}_eps${E}_s${S}_sanitized.jsonl" \
            "$V/finetunes/bert_${TAG}_eps${E}_s${S}.json" "$NL" "$S"
    done
}

losanr() {  # TAG DATASET K RETRIEVAL DELTA NUM_LABELS [POOL_TIER POOL_SIZE]
    local S
    for S in 0 1 2; do
        losanr_one "$1" "$2" "$3" "$4" "$5" "$6" "$S" "${7:-}" "${8:-0}"
    done
}

# PAIRS: resubmit single (variant, seed) cells, e.g.
#   PAIRS="imdb_k4_ctx|imdb|4|ctx|1e-5|2|2 amazon_k0_ctx|amazon|0|ctx|1e-7|2|0" SECTIONS=none
for P in ${PAIRS:-}; do
    f() { echo "$P" | cut -d'|' -f"$1"; }
    # optional fields 8 and 9: public pool tier and pool size
    losanr_one "$(f 1)" "$(f 2)" "$(f 3)" "$(f 4)" "$(f 5)" "$(f 6)" "$(f 7)" "$(f 8)" "$(f 9)"
done

# delta is 1e-5 for IMDB and AG News and 1e-7 for Amazon (see the paper's setup section)
if want A; then
    for K in 2 4 8 16 32 64 0; do losanr "imdb_k${K}_ctx" imdb "$K" ctx 1e-5 2; done
fi
if want B; then
    losanr amazon_k0_ctx amazon 0 ctx 1e-7 2
    losanr agnews_k0_ctx agnews 0 ctx 1e-5 4
fi
if want C; then
    losanr imdb_k16_nn imdb 16 nn 1e-5 2
    losanr imdb_k0_nn imdb 0 nn 1e-5 2
fi
if want D; then
    for T in imdb rotten_tomatoes amazon agnews; do
        losanr "imdb_k16_ctx_pool-${T}10k" imdb 16 ctx 1e-5 2 "$T" 10000
    done
fi
if want E; then
    for S in 0 1 2; do
        SAN=$(sbatch --parsable $ACCT --job-name="san-santext-s$S" --export=ALL,MODE=santext,NSEED=$S \
            protocol/slurm/run_sanitize_variant.sh)
        echo "san-santext-s$S $SAN" >> variant_jobs.txt
        for E in 2 2.25 2.5 2.75 3; do
            bert "$SAN" "bv-imdb_santext-e$E-s$S" imdb "$V/imdb_santext_eps${E}_s${S}_sanitized.jsonl" \
                "$V/finetunes/bert_imdb_santext_eps${E}_s${S}.json" 2 "$S"
        done
    done
fi
if want F; then
    # needs $V/imdb_dpbart_eps{E}_sanitized.jsonl from protocol/dpbart_to_jsonl.py
    for E in ${DPBART_EPS:-50 100 250 500 1000}; do
        for S in 0 1 2; do
            bert "" "bv-imdb_dpbart-e$E-s$S" imdb "$V/imdb_dpbart_eps${E}_sanitized.jsonl" \
                "$V/finetunes/bert_imdb_dpbart_eps${E}_s${S}.json" 2 "$S"
        done
    done
fi
if want G; then
    losanr imdb_k16_ctx imdb 16 ctx 1e-5 2
    losanr amazon_k16_ctx amazon 16 ctx 1e-7 2
    losanr agnews_k16_ctx agnews 16 ctx 1e-5 4
fi
echo "submitted $(( $(wc -l < variant_jobs.txt) - N0 )) jobs (sections: $SECTIONS)"
squeue -u "$USER" -h -o "%T %r" | sort | uniq -c
