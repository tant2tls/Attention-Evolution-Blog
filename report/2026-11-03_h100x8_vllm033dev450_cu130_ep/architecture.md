# Architecture inventory (calculated from checkpoint)

| model | total params (B) | active / token (B) | lookup tables (B) | layers | attention types | KV/state dtype | status |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| Llama 3.1 70B Instruct | 70.55 | 69.5 | 0.0 | 80 | GQA full attention: 80 | fp8 | calculated from checkpoint |
| DeepSeek V4 Flash 0731 | 304.18 | 13.27 | 0.0 | 43 | sliding-window MLA only (compress_ratio=0): 2; window + 4x compressed KV + indexer (compress_ratio=4): 21; window + 128x compressed KV (compress_ratio=128): 20 | fp8 | calculated from checkpoint |
| Qwen3.8 Flash-Next FP8 | 180.0 | 6.67 | 51.2 | 48 | Gated DeltaNet linear attention (recurrent): 36; QSA sparse full attention + indexer: 12 | fp8 | calculated from checkpoint |
| MiMo V2.6 Flash-MOPD | 310.76 | 14.82 | 0.0 | 48 | full attention (GQA, sinks): 9; sliding-window attention (128): 39 | auto | calculated from checkpoint |
| DeepSeek V4.1 Flash | 763.21 | 15.82 | 196.93 | 40 | sliding-window MLA only (compress_ratio=0): 2; window + uncompressed sparse (compress_ratio=1): 20; window + 2x compressed KV (compress_ratio=2): 18 | pending | calculated from checkpoint |
| GLM-5.3 Flash | 321.32 | 16.74 | 0.0 | 45 | KDA linear attention (recurrent): 34; DSA sparse MLA attention + indexer: 11 | pending | calculated from checkpoint |

Active = non-expert base weights (excluding input embedding and lookup tables) + lm_head + routed experts × experts_per_tok / n_routed. Prefill and decode use the same per-token definition. MTP/draft and vision/audio weights are excluded from active and listed in raw inventories. Packed 4-bit weights count 2 params per byte; quantization scales are excluded.
