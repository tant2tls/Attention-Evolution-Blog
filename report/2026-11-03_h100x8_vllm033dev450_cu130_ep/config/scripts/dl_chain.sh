#!/usr/bin/env bash
cd /root/Attention-Evolution-Blog
source bench/env.sh
unset HF_HUB_OFFLINE
S=report/2026-11-03_h100x8_vllm033dev450_cu130_ep
q() { for k in "$@"; do python bench/download_model.py --model-key $k --workers 32 >> $S/logs/download_chain.log 2>&1; done; }
q llama-31-70b qwen-38-fp8 v41 &
q v4-0731 mimo-v26 glm-53 &
wait
echo "ALL_DOWNLOADS_DONE" >> $S/logs/download_chain.log
