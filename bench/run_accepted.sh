#!/usr/bin/env bash
# Accepted session + matched unprobed control for one frozen model, sequentially, one model at a time.
# usage: bench/run_accepted.sh MODEL_KEY KV_DTYPE SESSION_SUFFIX EXPECTED_POINTS EXPECTED_OVERHEAD_POINTS
set -u
cd "$(dirname "$0")/.."
source bench/env.sh
K=$1; KV=$2; SUF=$3; EP=$4; EO=$5
python bench/session.py --model-key $K --session-id $K-$SUF --role "accepted" --expected-points $EP --timeout 7200 -- \
  python bench/attn_study.py --model-key $K --session-id $K-$SUF --kv-cache-dtype $KV --mode study
python bench/freeze_config.py --model-key $K --add-accepted $K-$SUF > /dev/null
python bench/session.py --model-key $K --session-id $K-overhead-$SUF --role "matched unprobed control" --expected-points $EO --timeout 7200 -- \
  python bench/attn_study.py --model-key $K --session-id $K-overhead-$SUF --kv-cache-dtype $KV --mode overhead
python bench/freeze_config.py --model-key $K --add-overhead $K-overhead-$SUF > /dev/null
echo "CHAIN_DONE $K $(date -u +%FT%TZ)"
