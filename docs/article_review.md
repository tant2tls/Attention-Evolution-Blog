# H100 article: evidence and editorial review

The user authorized refining the root article on 2026-11-05 while retaining its six sections.
The article now follows the attention journey: full GQA → local/global layers → compressed history →
recurrent state with sparse retrieval → shared long-range state. This is a design narrative, not a
release chronology or an architecture-only ranking. Seven original SVGs illustrate the mechanisms.
They are code-native diagrams, not measured buffer pictures. No author attribution is invented.

## Selected evidence

Only `report/2026-11-03_h100x8_vllm033dev450_cu130_ep/` supplies performance/storage values.
B200 is incomplete and excluded. Earlier cu126/cu130 BF16-Qwen sessions are historical references.
The H100 manifest pins one node/environment, vLLM commit `17e675965d54bef8a3eb77ea92c6bae195ef793c`,
torch `2.14.0+cu130`, six checkpoint revisions/config IDs, TP8 and EP for MoE.

The build checks all 1,106 entries currently in that study's hash ledger. It finds 138 supported
points, each with exactly three independent valid repetitions: 23 settled-memory, 92 decode and
23 prefill. Llama's six 128K grid configurations are unsupported. Displayed decode batches are one
and eight; batches two/four remain in the evidence. Summary medians are recomputed from repetition
CSVs and checked against `summary.csv`. Every included row must reference an existing raw file,
the selected immutable revision and its frozen config ID. Per-layer owner bytes reconcile to the
whole-model memory total at every supported context. The builder never edits the report or ledger.

## Interpretation that must survive edits

- State bars show occupied attention-state blocks at settled decode, summed across eight physical
  ranks, excluding weights/activations/reserved pools. They include compression/window rounding,
  indices/tails, recurrent buffers and physical replicas. Prompt labels and actual live lengths are
  both shown. The eight-request option is calculated scaling, never measured capacity.
- KV formats differ: MiMo fell back to BF16 after the FP8 failure. DeepSeek uses packed formats;
  Qwen/GLM keep mixed-precision recurrent/index/tail state. Identical FP8 flags do not establish an
  equal-precision comparison. FP8 scale warnings and selected configs remain available.
- Timing is instrumented CUDA-event intervals around attention modules, layer sums on each rank,
  followed by maximum rank. It is not whole-request latency, a distributed critical path, core-only
  attention, hardware traffic or summed kernel durations. Eager prefill can include host-launch gaps.
- Decode plots show three-repeat min/median/max. Probe sensitivity uses measured attention minus
  the positive whole-model slowdown from matched unprobed controls. It is an assumption-based
  sensitivity calculation, not a rigorous confidence interval or an exact corrected attention time.
- All claims refer to deployments, precisions, this build and workload. No answer-quality claim,
  hardware-bandwidth conclusion or universal speed ranking is made.
- GLM's interrupted session and successful resume use the same frozen config; the first failure
  and valid carried runs remain in the evidence. Its CUDA failure root cause is still open.
- V4.1 sharing links shown by the layer inspector come from config inference, not individually
  traced consumer edges. Executed prefill positions come from saved `prefill.csv` records.
- At 64K the article reports rounded medians: state/decode-b8/prefill-8K = Llama 10.01 GiB /
  8.68 ms / 358.1 ms; Qwen 3.50 GiB / 2.28 ms / 63.9 ms; V4.1 0.92 GiB / 3.93 ms / 72.1 ms.
  The numeric data file and builder are authoritative for precision and future regeneration.

## Build and review

```bash
python bench/build_attention_article.py
python bench/build_attention_article.py --check
```

Edit prose in `bench/attention_article_template.html`, interactions in `bench/attention_article.js`,
and figures in `bench/attention_diagrams.py`; regenerate the root page and SVG/data assets.
The static 64K table is rendered during the build and stays readable without JavaScript.
Chart and per-layer controls use embedded data, so the page needs no external scripts or fonts.
Keep `assets/attention/` alongside the page when embedding or publishing it on the lab website;
retain access to relative report links or rewrite those links to the evidence repository.

Browser review uses hidden Chrome: desktop 1440 px and phone 390 px; six architecture tabs,
all four memory contexts, request scaling, V4.1 shared references, decode batches 1/8,
all prefill endpoints, absent Llama 128K points, SVG exports, JavaScript exceptions and page overflow.
Final screenshots are inspected for diagram readability and chart layout. SVG diagrams/line plots
can scroll inside their containers on a narrow screen. Chart exports contain source/metric labels.
No measurements or publishing are part of the article build.

## Interactive storage composition (2026-11-05)

The storage section follows Kan's context-slider reference: select a model, move between measured
1K/16K/64K/128K anchors, and read the memory total plus component percentages. The primary view
always represents one decoding request. Default units are MiB per GPU (node bytes / 8); the node
view shows GiB. Percentages describe storage, not runtime or the fraction of attention layers.
No intermediate-context interpolation or capacity claim is made. Llama 128K stays unsupported.

Names follow the mechanism: `global_attn` for Llama/MiMo full KV, `swa_attn` for local windows,
`sparse_attn` / `sparse_mla` for Qwen/GLM retrieval state, `compressed_kv` / `shared_kv` for DeepSeek,
and `index_tail` for auxiliary buffers. The recurrent category is further partitioned into
`recurrent_state`, `sconv` and `state_padding` using recorded `MambaSpec.shapes`, dtypes, page size
and observed live-block counts. This is a calculated suballocation of measured occupied pages,
not new hardware measurements. It must reconcile to the original recurrent bytes for each context,
and every component must sum back to the saved total. Non-attention PLE buffers stay excluded.

Reference writing/layout reviewed: https://github.com/SyFI-ServingStudio/ServingStudioIntro/tree/main/content/blog/introducing-servingstudio
The article/index, metadata and asset listing were read; no authorship, numbers or imagery were copied.

## Explanation pass (2026-11-07)

Read Azamat Yelmagambetov’s “The Evolution of Attention: Unlocking LLMs’ Full Potential”:
https://medium.com/@ayelmagam/evolution-of-attention-a8c7e285bcaf
Used its teaching approach as inspiration: familiar example before notation, small conceptual steps,
and a problem/choice/trade-off progression. Wrote original explanations; copied no prose or images.
The six section titles, H100 data, controls and interpretation boundaries stay intact.
Q/K/V now precede the formula; prefill and decode are defined on first use. The variants distinguish
head sharing, windows, compression, sparse selection, recurrent updates and cross-layer ownership.
The memory section explains why layer percentages differ from byte percentages. Prefill explicitly
distinguishes an 8K chunk from whole-prompt latency. Analogies explain mechanics, not answer quality.
