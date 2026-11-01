# Attention architectures: inherited leads, not a new inventory

Adapted only from Frontier `target.md`, `docs/models.md` and the five `reproduce/<key>.md` guides.
These are historical observations, to revalidate against the deployed checkpoint and runtime.
No HTML parameter count or byte formula is accepted as verified. No new GPU data is supplied here.

## Checkpoint identities and attention leads

| Key | Checkpoint | Immutable revision | Attention inventory lead | Historical implementation lead |
| --- | --- | --- | --- | --- |
| `v4-0731` | `deepseek-ai/DeepSeek-V4-Flash-0731` | `7872f01b1d1fe23eabc4c98b48bffcef5a386062` | 43; recent window plus indexed compressed history | DeepseekV4ForCausalLM; fp8_ds_mla and FP8 index cache |
| `v41` | `deepseek-ai/DeepSeek-V4.1-Flash` | `dba1be0a40aa45a94ad051997016db3960a90277` | 40; shared compressed state and bounded decoder replay | DeepseekV41ForCausalLM; fp8_ds_mla; shared state ownership must be traced |
| `mimo-v26` | `XiaomiMiMo/MiMo-V2.6-Flash-MOPD` | `2479e2d0029eca9a34cc7e7f55a121925f81908e` | 48; 39 window + 9 full attention | MiMoV2OmniForCausalLM; FLASH_ATTN_DIFFKV; BF16 KV |
| `qwen-38-bf16` | `Qwen/Qwen3.8-Flash-Next` | `de4b8e4d43b917e7706784d8bb445c9af86a3540` | 48; 36 recurrent + 12 sparse history layers | Qwen4ExpForConditionalGeneration; GDN prefill; original BF16 weights |
| `glm-53` | `zai-org/GLM-5.3-Flash` | `eb9eb208eb0d988989d07a6a12d0fdeb5f52574a` | 45; 34 recurrent + 11 sparse attention | Glm5NextForConditionalGeneration; native FP8 weights; heterogeneous cache groups |
| `llama-31-70b` | Llama 3.1 70B Instruct; confirm exact repository and access | **Not yet pinned** | GQA baseline; verify all dimensions and layer counts | Not measured in Frontier; verify runtime support and context limit |

Freeze the five inherited pins consistently in the new manifest and runner. The generic V4 name
in the draft maps to the inherited **0731** checkpoint here; another revision is a different deployment.
Pin tokenizer and remote code as well as weights. Resolve Llama's repository/revision before any GPU launch.

## Per-model inspection that matters

- **V4 0731:** inspect window, compressed long-range state, indexer cache, compression/tail buffers
  and actual cache dtypes separately. Sparse attention core excludes substantial preparation work.
- **V4.1:** config has `kv_source_layer_ids` `[2, 8, 14, 20]` and window 128. Trace owners and aliases;
  never multiply a global shared budget by all 40 consumers. Runtime `554340f3` executed 21 full-chunk
  and 19 trimmed layers; `a98247ab4` executed 20 and 20. September `44af287e` had no such prefill skip.
  Prompt logprobs and small chunks can change the path. Record actual positions by layer.
  Engram tables used substantial pinned host RAM; they are weights, not attention state.
- **MiMo:** Frontier records full-attention layers **0, 5, 11, 17, 23, 29, 35, 41, 47** (zero based).
  The prototype uses a different schematic sequence. Revalidate the exact config rather than copying
  the HTML. Inspect separate key/value dimensions, learned sinks, local-window eviction and BF16 KV.
- **Qwen:** keep the original BF16 checkpoint. Separate recurrent state, sparse KV, index/tail state,
  and n-gram embedding weights in pinned host RAM. Large recurrent cache block sizes do not prove
  context-linear recurrent state. Confirm sparse-layer placement and runtime shapes.
- **GLM:** distinguish fixed recurrent state from the 11 growing sparse caches, indexer and page
  padding. Weight FP8 does not establish KV/recurrent dtype. No thinking-off switch was available
  in Frontier: `reasoning_effort=low` was used; other models disabled thinking.
- **Llama:** establish its own config/weight inventory, KV dtype, TP partition/replication and runtime
  path. Do not treat the prototype's 80-layer memory formula as collected evidence.

## Source-backed inventory to collect

Save checkpoint config and weight-index hashes; shapes/dtypes and unique parameter counts; tied-weight
aliases; base versus auxiliary/MTP versus conditional table weights; active parameter definitions for
prefill/decode; per-layer attention types; state owners/consumers; loaded classes and backend log lines.
Active parameters are neither resident bytes nor GPU speed. Missing validation stays `unverified`.

## Pinned config references

- [v4-0731: config at inherited revision](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/7872f01b1d1fe23eabc4c98b48bffcef5a386062/config.json)
- [v41: config at inherited revision](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/dba1be0a40aa45a94ad051997016db3960a90277/config.json)
- [mimo-v26: config at inherited revision](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-MOPD/blob/2479e2d0029eca9a34cc7e7f55a121925f81908e/config.json)
- [qwen-38-bf16: config at inherited revision](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/de4b8e4d43b917e7706784d8bb445c9af86a3540/config.json)
- [glm-53: config at inherited revision](https://huggingface.co/zai-org/GLM-5.3-Flash/blob/eb9eb208eb0d988989d07a6a12d0fdeb5f52574a/config.json)
