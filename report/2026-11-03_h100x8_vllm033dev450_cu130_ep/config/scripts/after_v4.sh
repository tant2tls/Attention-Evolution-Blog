#!/usr/bin/env bash
cd /root/Attention-Evolution-Blog
until grep -q CHAIN_DONE /root/attn-study/v4-chain.out; do sleep 5; done
source bench/env.sh
python bench/session.py --model-key qwen-38-fp8 --session-id qwen-fp8-smoke1 --role "validation fp8" --timeout 3600 -- \
  python bench/attn_study.py --model-key qwen-38-fp8 --session-id qwen-fp8-smoke1 --kv-cache-dtype fp8 --mode smoke
