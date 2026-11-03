#!/usr/bin/env bash
# Two parallel download queues in measurement order. Token read by huggingface_hub from $HF_HOME/token.
cd "$(dirname "$0")/../../../.."; source bench/env.sh; unset HF_HUB_OFFLINE
S=report/$ATTN_STUDY_ID
(for k in llama-31-70b qwen-38-fp8 v41; do python bench/download_model.py --model-key $k >> $S/logs/download_chain_a.out 2>&1; done) &
(for k in v4-0731 mimo-v26 glm-53; do python bench/download_model.py --model-key $k >> $S/logs/download_chain_b.out 2>&1; done) &
wait; echo "$(date -u +%FT%TZ) downloads done" >> $S/logs/download_chain_a.out
