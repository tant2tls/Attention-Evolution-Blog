#!/usr/bin/env bash
cd /root/Attention-Evolution-Blog
until [ -f /root/attn-study/v41-chain.out ] && grep -q CHAIN_DONE /root/attn-study/v41-chain.out; do sleep 5; done
source bench/env.sh
python bench/session.py --model-key glm-53 --session-id glm-fp8-smoke1 --role "validation fp8" --timeout 3600 -- \
  python bench/attn_study.py --model-key glm-53 --session-id glm-fp8-smoke1 --kv-cache-dtype fp8 --mode smoke
