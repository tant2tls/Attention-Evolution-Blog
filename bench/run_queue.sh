#!/usr/bin/env bash
# Two-phase unattended queue (attention_measure_plan.md section 7a). One model at a time, owned sessions.
#   Phase A  validate every model (FP8 first; one FP8 retry for intermittent failures; then `auto`) and freeze.
#   Phase B  accepted session (+ one resumed retry) and matched unprobed control, in REVERSE order of phase A
#            so the most recently loaded checkpoints are still in page cache.
# usage: bench/run_queue.sh SUFFIX [model ...]     (default: all six in study order)
# Writes a decision log to report/<study>/logs/queue_<SUFFIX>.log. Never pushes; never retries more than stated.
set -u
cd "$(dirname "$0")/.."
source bench/env.sh
SUF=$1; shift
MODELS=("$@"); [ ${#MODELS[@]} -eq 0 ] && MODELS=(llama-31-70b v4-0731 qwen-38-fp8 mimo-v26 v41 glm-53)
STUDY=$(python -c 'import sys; sys.path.insert(0,"bench"); import study_config as s; print(s.STUDY_DIR)')
QLOG=$STUDY/logs/queue_$SUF.log
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$QLOG"; }
expected() { python - "$1" <<'PY'
import sys; sys.path.insert(0, "bench"); import study_config as s
k = sys.argv[1]; mml = 131072 if s.MODELS[k]["extra"].get("max_model_len") == 131072 else 132096
ok = [L for L in s.CONTEXTS if L + -(-8 * L // 8192) + 108 <= mml]
nb = len(s.DECODE_BATCHES)
print((2 + nb) * len(ok), (1 + nb) * len(ok))   # accepted (memory+decode+prefill), control (decode+prefill)
PY
}
smoke() {  # model kv id -> 0 if the gate passed
  python bench/session.py --model-key "$1" --session-id "$3" --role "validation $2" --timeout 3600 -- \
    python bench/attn_study.py --model-key "$1" --session-id "$3" --kv-cache-dtype "$2" --mode smoke >/dev/null
}
declare -A KV
for K in "${MODELS[@]}"; do                                  # ---------------- phase A
  if [ -f "$STUDY/config/${K}__selected.json" ]; then log "A $K already frozen; skip"; KV[$K]=frozen; continue; fi
  tried=""
  for attempt in fp8:1 fp8:2 auto:1; do
    kv=${attempt%%:*}; n=${attempt##*:}; sid="$K-$kv-smoke-$SUF-$n"; tried="$tried,$sid"
    log "A $K validate kv=$kv session=$sid"
    if smoke "$K" "$kv" "$sid"; then
      reason=""; [ "$kv" != fp8 ] && reason="fp8 validation failed twice (${tried#,}); see their logs and fixes.md"
      python bench/freeze_config.py --model-key "$K" --validated-session "$sid" --kv-cache-dtype "$kv" \
        --attempts "${tried#,}" ${reason:+--fallback-reason "$reason"} >/dev/null
      KV[$K]=$kv; log "A $K frozen kv=$kv"; break
    fi
    log "A $K validation failed: $sid"
  done
  [ -z "${KV[$K]:-}" ] && { KV[$K]=none; log "A $K NO supported runnable choice found; document before going on"; }
done
for (( i=${#MODELS[@]}-1; i>=0; i-- )); do                   # ---------------- phase B (reverse order)
  K=${MODELS[$i]}; [ "${KV[$K]}" = none ] && { log "B $K skipped (not validated)"; continue; }
  kv=$(python -c "import json;print(json.load(open('$STUDY/config/${K}__selected.json'))['kv_dtype_requested'])")
  read EP EO <<<"$(expected "$K")"
  sid="$K-$SUF"; log "B $K accepted $sid (kv=$kv, expect $EP points)"
  python bench/session.py --model-key "$K" --session-id "$sid" --role accepted --expected-points "$EP" --timeout 7200 -- \
    python bench/attn_study.py --model-key "$K" --session-id "$sid" --kv-cache-dtype "$kv" --mode study >/dev/null
  rc=$?; python bench/freeze_config.py --model-key "$K" --add-accepted "$sid" >/dev/null
  if [ $rc -ne 0 ]; then                                     # one resumed retry, carrying valid (point, rep)s
    log "B $K accepted failed (rc=$rc); resumed retry $sid-r"
    python bench/session.py --model-key "$K" --session-id "$sid-r" --role "accepted retry (resume $sid)" \
      --expected-points "$EP" --timeout 7200 -- python bench/attn_study.py --model-key "$K" --session-id "$sid-r" \
      --kv-cache-dtype "$kv" --mode study --resume-sessions "$sid" >/dev/null
    rc=$?; python bench/freeze_config.py --model-key "$K" --add-accepted "$sid-r" >/dev/null
    [ $rc -ne 0 ] && log "B $K retry failed too (rc=$rc): document in fixes.md; continuing with control"
  fi
  log "B $K control $K-overhead-$SUF"
  python bench/session.py --model-key "$K" --session-id "$K-overhead-$SUF" --role "matched unprobed control" \
    --expected-points "$EO" --timeout 7200 -- python bench/attn_study.py --model-key "$K" \
    --session-id "$K-overhead-$SUF" --kv-cache-dtype "$kv" --mode overhead >/dev/null
  python bench/freeze_config.py --model-key "$K" --add-overhead "$K-overhead-$SUF" >/dev/null
  python bench/analyze.py >/dev/null && log "B $K done; CSVs regenerated (charts/README/push are manual)"
done
log "QUEUE_DONE"
