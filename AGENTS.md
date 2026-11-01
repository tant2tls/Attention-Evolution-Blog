# Attention measurement first

Read `README.md`, `attention_measure_plan.md`, `docs/model_architectures.md`,
`docs/gpu_setup_and_fixes.md`, `docs/README.md`, `report/README.md` and `bench/README.md` before changing measurement code.

- The active task is collecting trustworthy real GPU evidence for six checkpoints: Llama 3.1 70B Instruct,
  DeepSeek V4 Flash 0731, V4.1 Flash, MiMo V2.6 Flash-MOPD, original BF16 Qwen3.8 Flash-Next,
  and native FP8 GLM-5.3 Flash. Freeze Llama's immutable revision before launching it.
- `attention_measure_plan.md` is the single plan. Only architecture inventory, attention-state memory,
  decode attention timing and prefill attention timing belong in this study. Parameter inventory is
  supporting architecture work; do not add MoE/FFN performance experiments.
- `index.html` contains synthetic timings, formulas and schematic layouts. Keep it as a prototype.
  Do not import its numbers into results or write/redesign the final blog yet.
- Prepare code and offline checks now. Run GPU work only on a user-provided, explicitly authorized
  server/session; this file transfer does not authorize rentals, downloads of weights or measurements.
- Target one 8 x H100 80GB node, TP8, EP disabled, utilization 0.82, one model at a time.
  Prefix caching and speculation/MTP off. Preserve native precision. Confirm supported settings
  from the installed runtime and server logs; document deviations instead of silently falling back
  to Frontier's TP8+EP / 0.90 settings.
- Install using `bash install.sh`: the latest nightly observed on 2026-11-02 is pinned to vLLM
  `0.33.0rc1.dev450+g17e675965`, commit `17e675965d54bef8a3eb77ea92c6bae195ef793c`.
  Archive the environment's `vllm-pin.txt` and `requirements.freeze.txt`, plus node preflight
  GPU/driver/CUDA records, with the study. Never silently
  upgrade to moving nightly or substitute the historical Frontier build. Validate six-model support
  before freezing accepted measurement runs; any required runtime change starts a new environment.
- One fresh study identity, one frozen runtime/environment, immutable model/tokenizer/remote-code
  revisions, shared public workload and recorded actual token counts. Never combine historical
  Frontier measurements or different builds as fresh controls.
- Count unique state owners once per physical storage region per rank. Record shared consumers,
  fixed recurrent state, index/tail buffers and window state. Keep logical live bytes, physical live
  block bytes, reserved pools, temporary workspace and total device allocation separate.
- Measure attention regions on GPU with cross-stream dependencies accounted for; distinguish
  attention core, projections/indexer/state work and whole-model elapsed time. Kernel sums and
  client timings are diagnostics, not attention-region elapsed time. Missing isolation means
  `unavailable`, never a synthetic substitute.
- Verify actual decode batch, context at each measured step, no mixed prefill, executed prefill
  positions per layer, warmup and three valid independent repetitions. Keep every failed attempt.
- Copied trace tools are historical offline aids. They do not implement the required six-model runners,
  exact live-state ownership, safe lifecycle, or validated attention timing. Read their limitations.
- Save all study outputs under `report/<study-id>/`: raw repetitions, CSVs, summaries, charts,
  manifests, `config/`, `logs/`, `model_completion.csv`, and `fixes.md` if bugs occur. No separate
  top-level results/charts output folders. Follow `report/README.md` for the exact structure.
- Every supported memory/decode/prefill point needs three independent valid repetitions with
  rebuilt state/requests. Preserve every raw run and failure; failures never count toward three.
  Summaries show median, min/max, valid count and raw IDs. Fewer than three means incomplete.
- Save each model session's UTC start, measurement completion, verified shutdown, terminal time,
  monotonic elapsed duration and status in `model_completion.csv`; show these in the study README.
  Append resumed/retried sessions without overwriting previous rows. Never mark a partial run complete.
- Keep reusable setup/download improvements and working experience in `docs/`, with version,
  exact steps and evidence links to the study. Study-specific bugs/fixes stay in the study's
  `fixes.md`; promote the general lesson into docs after verification.
- Preserve raw evidence byte-for-byte. Generate summaries and charts from validated raw data,
  with manifests and hashes. Use explicit `measured`, `calculated from checkpoint`, `unverified`,
  `unsupported`, `failed`, or `unavailable` labels. Never report a missing value as zero.
- Store credentials outside Git. Keep weights, caches, environments and large traces out of Git;
  preserve raw traces on the node or an authorized artifact location with hashes and retrieval notes.
  Use study-owned paths; never delete shared caches or kill unowned GPU processes.
- Own the server process group; stop it on success, failure, interruption and timeout. Verify owned
  workers exit and GPU usage returns to the recorded idle baseline before the next launch.
- Do not commit, push, publish or merge unless requested. Finish with raw evidence, reproducible
  CSVs, three charts, architecture table and an honest account of missing measurements.
