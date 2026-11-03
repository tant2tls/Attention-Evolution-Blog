#!/usr/bin/env bash
# B200 study driver: one model at a time (validate+freeze, accepted, control via bench/run_queue.sh),
# gated on that model's download, so each model's results can be pushed as soon as it finishes.
# Same per-model policy as run_queue.sh; only the cross-model order differs (no phase A for all six first).
set -u
cd "$(dirname "$0")/../../../.."
export ATTN_STUDY_ID=2026-11-03_b200x8_vllm033dev450_cu130_ep ATTN_DECODE_BATCHES=1,8
export ATTN_NODE_DESC="8 × B200 180GB" ATTN_DRIVER=580.105.08
source bench/env.sh
S=report/$ATTN_STUDY_ID; D=$S/logs/driver.log
log() { echo "$(date -u +%FT%TZ) $*" >> "$D"; }
ALL="llama-31-70b v4-0731 qwen-38-fp8 mimo-v26 v41 glm-53"
# Rule: no download beside accepted timing. Wait until every download has a final status before any launch.
log "WAIT_ALL_DOWNLOADS"
for K in $ALL; do until [ -f "$S/logs/download_$K.json" ] && ! grep -q '"running"' "$S/logs/download_$K.json"; do sleep 10; done; done
pgrep -f bench/download_model.py >/dev/null && { log "DOWNLOAD_PROCESS_STILL_RUNNING"; while pgrep -f bench/download_model.py >/dev/null; do sleep 5; done; }
log "ALL_DOWNLOADS_FINISHED"
for K in $ALL; do
  log "WAIT_DOWNLOAD $K"
  until st=$(python -c "import json;print(json.load(open('$S/logs/download_$K.json'))['status'])" 2>/dev/null) && [ "$st" != running ]; do sleep 30; done
  if [ "$st" != ok ]; then log "DOWNLOAD_FAILED $K status=$st"; continue; fi
  python3 bench/checkpoint_inventory.py --model-key $K >> $S/logs/inventory_$K.out 2>&1 || log "INVENTORY_FAILED $K"
  log "START $K"
  bench/run_queue.sh s1 $K >> $S/logs/queue_s1.out 2>&1
  log "QUEUE_RC $K $?"
  python3 bench/analyze.py >> $S/logs/report_$K.out 2>&1 || log "ANALYZE_FAILED $K"
  /root/attn-study/venv-report/bin/python bench/make_report.py >> $S/logs/report_$K.out 2>&1 || log "CHARTS_FAILED $K"
  python3 bench/write_readme.py >> $S/logs/report_$K.out 2>&1 || log "README_FAILED $K"
  log "MODEL_DONE $K"
done
log "ALL_DONE"
