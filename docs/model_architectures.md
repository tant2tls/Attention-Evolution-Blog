# Attention architectures: inherited leads, not a new inventory

Originally adapted from Frontier `target.md`, `docs/models.md` and five per-model guides.
Qwen was subsequently replaced by the official FP8 checkpoint on 2026-11-02; its source is linked below.
These are historical observations, to revalidate against the deployed checkpoint and runtime.
No HTML parameter count or byte formula is accepted as verified. No new GPU data is supplied here.

## Checkpoint identities and attention leads

| Key | Checkpoint | Immutable revision | Attention inventory lead | Historical implementation lead |
| --- | --- | --- | --- | --- |
| `v4-0731` | `deepseek-ai/DeepSeek-V4-Flash-0731` | `7872f01b1d1fe23eabc4c98b48bffcef5a386062` | 43; recent window plus indexed compressed history | DeepseekV4ForCausalLM; fp8_ds_mla and FP8 index cache |
| `v41` | `deepseek-ai/DeepSeek-V4.1-Flash` | `dba1be0a40aa45a94ad051997016db3960a90277` | 40; shared compressed state and bounded decoder replay | DeepseekV41ForCausalLM; fp8_ds_mla; shared state ownership must be traced |
| `mimo-v26` | `XiaomiMiMo/MiMo-V2.6-Flash-MOPD` | `2479e2d0029eca9a34cc7e7f55a121925f81908e` | 48; 39 window + 9 full attention | MiMoV2OmniForCausalLM; FLASH_ATTN_DIFFKV; BF16 KV |
| `qwen-38-fp8` | `Qwen/Qwen3.8-Flash-Next-FP8` | `236dfdf285828023ca3bcd3f37366c58a3469b13` | 48; 36 recurrent + 12 sparse history layers | Config class Qwen4ExpForConditionalGeneration; official mixed FP8/BF16 weights; executed backend/state dtypes unverified |
| `glm-53` | `zai-org/GLM-5.3-Flash` | `eb9eb208eb0d988989d07a6a12d0fdeb5f52574a` | 45; 34 recurrent + 11 sparse attention | Glm5NextForConditionalGeneration; native FP8 weights; heterogeneous cache groups |
| `llama-31-70b` | Llama 3.1 70B Instruct; confirm exact repository and access | **Not yet pinned** | GQA baseline; verify all dimensions and layer counts | Not measured in Frontier; verify runtime support and context limit |

Freeze the four unchanged Frontier pins and the new Qwen FP8 pin consistently in the new manifest and runner. The generic V4 name
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
- **Qwen:** use the official FP8 checkpoint at `236dfdf285828023ca3bcd3f37366c58a3469b13`. Its config declares
  `quant_method=fp8`, dynamic activations and `[128, 128]` weight blocks; text dtype is BF16.
  Attention/recurrent projections appear in `modules_to_not_convert`, so this is mixed precision,
  not every tensor in FP8. Separate routed weights, n-gram weights, recurrent state, sparse KV and
  index/tail buffers. Confirm their loaded dtypes, device placement, backend and runtime shapes.
  The config retains 48 layers (36 linear-attention and 12 full-attention entries); those config
  names alone do not establish every executed attention operation.
  Frontier's BF16 logs are historical leads only, not execution evidence for this FP8 deployment.
  Expected weight-residency savings do not establish savings in attention-state bytes.
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

## Current runtime policy (2026-11-03)

Read [experience.md](experience.md). All launches request TP8 + EP on (dense Llama has no experts),
utilization 0.82 and FP8 KV first. Record a failed FP8 attempt before fallback to `auto` or another
supported runnable choice; preserve resolved formats and actual buffer dtypes. The recorded Qwen
FP8-KV validation used BF16 weights, so revalidate the selected FP8 checkpoint. MiMo FP8-KV success
and GLM startup are not established by the historical notes. V4.1's EP-off TP8 launch failed; GLM's
failure cause is still unknown. Do not treat these observations as new measurements.

## Precision and comparison policy

Preserve each checkpoint's native mixed precision. Report actual weight, KV/latent, index/tail,
recurrent and scale dtypes separately, including bytes/owner and per-rank replication. Show both
actual storage and structural shapes/ownership, so lower precision is not mistaken for an
architecture improvement. Qwen FP8 may reduce weight residency and change speed/backend, but
attention-state and elapsed-time differences must be measured. No BF16-versus-FP8 ablation is added.

## Pinned config references

- [v4-0731: config at inherited revision](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/7872f01b1d1fe23eabc4c98b48bffcef5a386062/config.json)
- [v41: config at inherited revision](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/dba1be0a40aa45a94ad051997016db3960a90277/config.json)
- [mimo-v26: config at inherited revision](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-MOPD/blob/2479e2d0029eca9a34cc7e7f55a121925f81908e/config.json)
- [qwen-38-fp8: official config at selected revision](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8/blob/236dfdf285828023ca3bcd3f37366c58a3469b13/config.json)
- [glm-53: config at inherited revision](https://huggingface.co/zai-org/GLM-5.3-Flash/blob/eb9eb208eb0d988989d07a6a12d0fdeb5f52574a/config.json)
