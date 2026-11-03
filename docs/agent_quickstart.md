# Agent quickstart: run the six-model attention study fast

For an agent that has been **explicitly authorized** to use an 8 × H100 80GB node. Rules still come from
[AGENTS.md](../AGENTS.md) and [the plan](../attention_measure_plan.md); this page is the shortest path that
follows them. Last full run: [study 2026-11-03](../report/2026-11-03_h100x8_vllm033dev450_cu130_ep/README.md)
(exact commands: [run_commands.md](../report/2026-11-03_h100x8_vllm033dev450_cu130_ep/config/run_commands.md)).

**First check [plan §0](../attention_measure_plan.md): the blog's GPU numbers already exist. Run GPU
work only for a listed rerun trigger, and then with the minimal grid `export ATTN_DECODE_BATCHES=1,8`.**

Expected time on the same build with the JIT archive: **~15 min setup + ~2–2.5 h unattended GPU queue**.
Without the archive add ~50 min of first-launch compilation.

## 0. Before anything (2 min)

```bash
cd <repo>
nvidia-smi; nvidia-smi --query-compute-apps=pid,used_memory --format=csv   # must be idle; never kill unowned work
df -h; findmnt -T /root -o FSTYPE; findmnt -T /var/tmp -o FSTYPE         # weights need ~1.5 TB; JIT caches on LOCAL disk
cat /sys/fs/cgroup/memory.max; free -g                                   # V4.1 pins ~190 GB host RAM
```

- Pick a **new study id** for a new node/runtime: `export ATTN_STUDY_ID=<date>_h100x8_vllm033dev450_cu130_ep`.
  Reuse the old id only to resume the same study on the same build.
- HF token: write it to `$ATTN_WORK_ROOT/hf/token` (`chmod 600`), never into the repo or logs. Llama is gated.
- Check push access early: `GIT_TERMINAL_PROMPT=0 git push --dry-run origin main`. Push only if the user asked.

## 1. Environment and downloads (≈15 min, mostly in background)

```bash
VLLM_VENV=$PWD/.venv-attention bash install.sh          # skip if the venv exists with the same vllm-pin.txt
source bench/env.sh                                      # HF_HOME, node-local JIT caches, restores JIT archive
S=report/$ATTN_STUDY_ID; mkdir -p $S/{config,logs,raw,charts}
cp install.sh .venv-attention/{vllm-pin.txt,requirements.freeze.txt} $S/config/
# downloads: two parallel queues in measurement order (see config/scripts/dl_chain.sh of the last study)
(unset HF_HUB_OFFLINE; for k in llama-31-70b qwen-38-fp8 v41; do python bench/download_model.py --model-key $k; done) &
(unset HF_HUB_OFFLINE; for k in v4-0731 mimo-v26 glm-53; do python bench/download_model.py --model-key $k; done) &
```

While downloads run (no GPU needed):

```bash
for k in llama-31-70b v4-0731 qwen-38-fp8 mimo-v26 v41 glm-53; do python3 bench/checkpoint_inventory.py --model-key $k; done
```

Gate: every `logs/download_<k>.json` has `"status": "ok"` before that model's launch.

## 2. One command for all GPU work

```bash
setsid nohup bench/run_queue.sh s1 > $S/logs/queue_s1.out 2>&1 < /dev/null &
```

- **Phase A** validates and freezes all six (FP8 → FP8 retry → `auto`), Llama first.
- **Phase B** runs accepted session + matched unprobed control per model in reverse order, with one
  resumed retry (`--resume-sessions`) that keeps valid repetitions.
- Decisions go to `$S/logs/queue_s1.log`. Watch it with a Monitor/until-loop on that file; do not poll with sleeps.
- Per-model gates are content checks inside the scripts: policy gate (`POLICY_ERRORS []`), expected attention
  regions on every rank, valid replay events, exact pure-decode batch, final-chunk step found.

Per-model timings to expect (warm): init 1–2 min, accepted 6–10 min, control 6–11 min.
Cold first launch compiles for minutes with GPUs idle and many `cc1plus`/`cicc` processes: that is normal
(Qwen ~16 min, V4 0731 ~12 min); the session timeout is 1 h.

## 3. After each model finishes (5 min)

```bash
python3 bench/analyze.py
/root/attn-study/venv-report/bin/python bench/make_report.py     # matplotlib venv, never the frozen env
python3 bench/write_readme.py
git add bench docs report && git commit -m "<model>: ..." && git push origin main   # only if authorized
```

Check before committing: `summary.csv` rows for the model are `complete` (or explained `unsupported`),
`model_completion.csv` has a row per session with verified shutdown, and nothing in the repo contains the token
(`grep -r hf_ . --exclude-dir=.venv-attention --exclude-dir=.git`).

## 4. When something fails

| Symptom | Do this |
| --- | --- |
| Any failure | Keep everything (logs, raw, completion row). Write `fixes.md` entry: symptom, cause, fix, verification, rerun needed. New session id for the retry. |
| `FileNotFoundError` in Triton cache, `CompiledKernel has no attribute module` | JIT cache is on a network/FUSE filesystem; fix `ATTN_JIT_ROOT` (fixes B3) |
| `Number of experts ... must be greater than 0 when expert parallelism is enabled` | Dense model: EP is N/A (already set for Llama, B2) |
| FP8 KV startup assertion (MiMo `fp8_kv_dequant requires fp16 or bf16 Q`) | Policy fallback to `auto`; record reason (B7). Do not force backends |
| Bare `CUDA error: invalid argument`, worker died (GLM) | Retry once with `--resume-sessions <failed session>` (B8); if it repeats, stop and report |
| `POLICY_ERRORS [...]` / regions ≠ expected | Do not measure. Check `raw/<k>/sessions/<id>/probe_info.json` for the resolved class and layers |
| Session stuck, GPUs 0 %, no compiler processes | Let the session timeout clean up; never `pkill -f` patterns (it can kill your own shell) |

## 5. Do not

- Run two models at once, download or compile beside accepted timing, or change utilization/EP/KV policy silently.
- Count failed attempts, smoke runs or carried repetitions twice; analysis only reads sessions registered in
  `config/<k>__selected.json`.
- Overwrite raw evidence or completion rows; append corrections.
- Install anything into the frozen venv (use the separate report venv for charts).
- Delete the downloaded weights or shared caches unless the user asks.

After the study, archive the JIT cache for the next run:
`tar -cf $ATTN_WORK_ROOT/jit-archive/cu13_vllm033dev450_g17e675965.tar -C $ATTN_JIT_ROOT cu13`.
