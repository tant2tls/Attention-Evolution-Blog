# Architecture inventory (calculated from checkpoint)

| model | total params (B) | active / token (B) | lookup tables (B) | layers | attention types | KV/state dtype | status |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| Llama 3.1 70B Instruct | 70.55 | 69.5 | 0.0 | 80 | GQA full attention: 80 | fp8 | calculated from checkpoint |
| DeepSeek V4 Flash 0731 | 304.18 | 13.27 | 0.0 | 43 | sliding-window MLA only (compress_ratio=0): 2; window + 4x compressed KV + indexer (compress_ratio=4): 21; window + 128x compressed KV (compress_ratio=128): 20 | fp8_ds_mla | calculated from checkpoint |
| qwen-38-fp8 | unverified | | | | | | unverified |
| mimo-v26 | unverified | | | | | | unverified |
| v41 | unverified | | | | | | unverified |
| glm-53 | unverified | | | | | | unverified |

Active = non-expert base weights (excluding input embedding and lookup tables) + lm_head + routed experts × experts_per_tok / n_routed. Prefill and decode use the same per-token definition. MTP/draft and vision/audio weights are excluded from active and listed in raw inventories. Packed 4-bit weights count 2 params per byte; quantization scales are excluded.
