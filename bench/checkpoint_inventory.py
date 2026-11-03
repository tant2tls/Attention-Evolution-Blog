"""Source-backed parameter and attention-layer inventory from a downloaded checkpoint (no GPU).

Reads every safetensors header (shapes/dtypes only) plus config.json at the pinned revision.
Outputs raw/<model>/architecture/{inventory.json, config.json copy, weight_index_sha256}.

Parameter rules (plan section 3):
* scale tensors (names containing "scale", dtype F8_E8M0) are quantization metadata, not parameters;
* packed 4-bit tensors (I8/U8 non-scale weights) count 2 parameters per stored byte;
* categories: routed experts, shared experts, input embeddings, lm_head, n-gram/Engram lookup tables,
  MTP/next-token/draft modules, vision/audio towers, other base-model weights;
* tied weights are counted once (identical tensor names only appear once in safetensors);
* active per token (decode and prefill alike under a per-token definition) =
  base non-expert weights excluding lookup tables and the input embedding
  + lm_head + routed experts x (experts_per_tok / n_routed_experts). Lookup tables are gathers,
  not GEMMs, and are reported separately. Labelled "calculated from checkpoint".
"""
from __future__ import annotations

import argparse
import collections
import glob
import hashlib
import json
import re
import shutil
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import study_config as sc  # noqa: E402

BYTES = {"F64": 8, "F32": 4, "BF16": 2, "F16": 2, "F8_E4M3": 1, "F8_E5M2": 1, "F8_E8M0": 1,
         "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1}


def category(name: str, layer_limit: int) -> str:
    n = name.lower()
    if any(s in n for s in ("visual", "vision", "vit.", "aligner", "audio", "patch_embed", "mm_projector")):
        return "vision_audio"
    if "dspark" in n or "dflash" in n:
        return "draft"
    if "mtp" in n or "nextn" in n:
        return "mtp"
    m = re.search(r"layers\.(\d+)\.", n)
    if m and int(m.group(1)) >= layer_limit:
        return "mtp"
    if "engram" in n or "ngram" in n:
        return "lookup_table"
    if "shared_expert" in n or "shared_experts" in n:
        return "shared_experts"
    if ".experts." in n or n.endswith(".experts") or "experts.w" in n or ".experts_" in n:
        return "routed_experts"
    if "lm_head" in n or n in ("head.weight",):
        return "lm_head"
    if "embed" in n:
        return "embedding"
    return "base_other"


def attention_inventory(key: str, tc: dict) -> dict:
    L = tc["num_hidden_layers"]
    if key == "llama-31-70b":
        return dict(types={"GQA full attention": L}, evidence="num_hidden_layers, num_key_value_heads",
                    detail=dict(q_heads=tc["num_attention_heads"], kv_heads=tc["num_key_value_heads"],
                                head_dim=tc["hidden_size"] // tc["num_attention_heads"]))
    if key in ("v4-0731", "v41"):
        cr = tc["compress_ratios"][:L]
        c = collections.Counter(cr)
        names = {0: "sliding-window MLA only", 4: "window + 4x compressed KV + indexer",
                 128: "window + 128x compressed KV", 2: "window + 2x compressed KV", 1: "window + uncompressed sparse"}
        types = {f"{names.get(k, f'ratio {k}')} (compress_ratio={k})": v for k, v in sorted(c.items())}
        det = dict(compress_ratios=cr, sliding_window=tc.get("sliding_window"), head_dim=tc.get("head_dim"),
                   kv_heads=tc.get("num_key_value_heads"), index_topk=tc.get("index_topk"))
        for k in ("kv_source_layer_ids", "index_source_layer_ids", "candidate_source_layer_id", "engram_layer_ids"):
            if k in tc:
                det[k] = tc[k]
        return dict(types=types, evidence="compress_ratios[:num_hidden_layers], sliding_window", detail=det)
    if key == "mimo-v26":
        p = tc["hybrid_layer_pattern"][:L]
        return dict(types={"full attention (GQA, sinks)": p.count(0), "sliding-window attention (128)": p.count(1)},
                    evidence="hybrid_layer_pattern (0=full, 1=SWA)",
                    detail=dict(full_layers=[i for i, x in enumerate(p) if x == 0],
                                kv_heads=tc["num_key_value_heads"], swa_kv_heads=tc.get("swa_num_key_value_heads"),
                                head_dim=tc["head_dim"], v_head_dim=tc.get("v_head_dim"),
                                sliding_window=tc.get("sliding_window")))
    if key == "qwen-38-fp8":
        lt = tc.get("layer_types")
        if lt:
            c = collections.Counter(lt[:L])
            ev = "layer_types"
        else:
            iv = tc["full_attention_interval"]
            full = sum(1 for i in range(L) if (i + 1) % iv == 0)
            c = {"full_attention": full, "linear_attention": L - full}
            ev = "full_attention_interval"
        return dict(types={"Gated DeltaNet linear attention (recurrent)": c.get("linear_attention", 0),
                           "QSA sparse full attention + indexer": c.get("full_attention", 0)},
                    evidence=ev, detail=dict(kv_heads=tc["num_key_value_heads"], head_dim=tc["head_dim"],
                                             indexer_budget=tc.get("indexer_budget"),
                                             indexer_compress_ratio=tc.get("indexer_compress_ratio")))
    if key == "glm-53":
        c = collections.Counter(tc["layer_types"][:L])
        return dict(types={"KDA linear attention (recurrent)": c.get("linear_attention", 0),
                           "DSA sparse MLA attention + indexer": c.get("deepseek_sparse_attention", 0)},
                    evidence="layer_types", detail=dict(kv_lora_rank=tc.get("kv_lora_rank"),
                                                        index_topk=tc.get("index_topk"),
                                                        linear_heads=tc["linear_attn_config"]["num_heads"]))
    return dict(types={}, evidence="unverified")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-key", required=True)
    ap.add_argument("--study-dir", default=str(sc.STUDY_DIR))
    a = ap.parse_args()
    key = a.model_key
    m = sc.MODELS[key]
    ck = sc.weights_dir(key)
    out = Path(a.study_dir) / "raw" / key / "architecture"
    out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((ck / "config.json").read_text())
    tc = cfg.get("text_config", cfg)
    if not (out / "config.json").exists():
        shutil.copy(ck / "config.json", out / "config.json")
    L = tc["num_hidden_layers"]
    cats = collections.defaultdict(lambda: dict(params=0, bytes=0, tensors=0))
    dtypes = collections.defaultdict(lambda: dict(params=0, bytes=0))
    scale_bytes = 0
    packed = 0
    files = sorted(glob.glob(str(ck / "**/*.safetensors"), recursive=True))
    files = [f for f in files if not any(Path(f).match(p) for p in m["ignore"])]
    for f in files:
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            hdr = json.loads(fh.read(n))
        for name, v in hdr.items():
            if name == "__metadata__":
                continue
            numel = 1
            for s in v["shape"]:
                numel *= s
            nbytes = numel * BYTES.get(v["dtype"], 1)
            if "scale" in name.lower() or v["dtype"] == "F8_E8M0":
                scale_bytes += nbytes
                continue
            params = numel
            if v["dtype"] in ("I8", "U8"):
                params = 2 * numel
                packed += numel
            c = category(name, L)
            cats[c]["params"] += params
            cats[c]["bytes"] += nbytes
            cats[c]["tensors"] += 1
            dtypes[v["dtype"] + ("(packed fp4)" if v["dtype"] in ("I8", "U8") else "")]["params"] += params
            dtypes[v["dtype"] + ("(packed fp4)" if v["dtype"] in ("I8", "U8") else "")]["bytes"] += nbytes
    total = sum(c["params"] for c in cats.values())
    n_routed = tc.get("n_routed_experts") or tc.get("num_experts")
    topk = tc.get("num_experts_per_tok")
    routed = cats["routed_experts"]["params"] if "routed_experts" in cats else 0
    frac = (topk / n_routed) if (n_routed and topk and routed) else 0.0
    active = (cats["base_other"]["params"] + cats["shared_experts"]["params"] + cats["lm_head"]["params"]
              + routed * frac)
    base_total = total - sum(cats[c]["params"] for c in ("mtp", "vision_audio", "draft") if c in cats)
    idx = ck / "model.safetensors.index.json"
    inv = dict(
        model=key, repo=m["repo"], revision=m["revision"], architectures=cfg.get("architectures"),
        config_sha256=hashlib.sha256((ck / "config.json").read_bytes()).hexdigest(),
        weight_index_sha256=hashlib.sha256(idx.read_bytes()).hexdigest() if idx.exists() else None,
        n_safetensors=len(files), total_params=total, base_model_params=base_total,
        categories={k: dict(v) for k, v in cats.items()}, dtypes={k: dict(v) for k, v in dtypes.items()},
        scale_bytes_excluded=scale_bytes, packed_fp4_bytes=packed,
        n_routed_experts=n_routed, experts_per_tok=topk, active_routed_fraction=frac,
        active_params_per_token=active,
        active_definition="base_other + shared_experts + lm_head + routed_experts*experts_per_tok/n_routed; "
                          "excludes input embedding, lookup tables, MTP/draft, vision/audio",
        layers=L, attention=attention_inventory(key, tc),
        quantization_config=cfg.get("quantization_config", tc.get("quantization_config")),
        label="calculated from checkpoint",
    )
    (out / "inventory.json").write_text(json.dumps(inv, indent=1, default=str))
    print(json.dumps({k: inv[k] for k in ("model", "total_params", "base_model_params", "active_params_per_token")}),
          {k: round(v["params"] / 1e9, 3) for k, v in inv["categories"].items()}, inv["attention"]["types"])


if __name__ == "__main__":
    main()
