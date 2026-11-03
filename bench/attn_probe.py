"""Worker extension: GPU attention-region timing and state-buffer introspection.

Loaded through vLLM ``worker_extension_cls`` = ``attn_probe.AttnProbeExt``. Importing this module
inside each worker (before the model is built) patches ``Worker.load_model``, ``execute_model``
and ``sample_tokens``:

* after load, every decoder layer's attention child (``attn`` / ``self_attn`` / ``linear_attn``)
  and the top-level model ``forward`` / ``compute_logits`` are wrapped with
  ``torch.cuda.Event(enable_timing=True, external=True)`` start/end records. The same event
  objects are reused for every eager call and every captured graph, so on CUDA-graph replay the
  event-record nodes re-record them. After each real step the worker synchronizes and reads the
  elapsed times relative to the model-forward start; an event that was not re-recorded in this
  step has a negative offset (or is outside the window) and the step is marked invalid.
* ``ATTN_PROBE_LAYERS=0`` keeps only the model/logits windows (matched unprobed control).

Every RPC returns errors as data; hooks never raise into the engine (experience F3).
"""
from __future__ import annotations

import os
import time
import traceback
from typing import Any

import torch

PROBE_LAYERS = os.environ.get("ATTN_PROBE_LAYERS", "1") != "0"
ATTN_CHILD_NAMES = ("attn", "self_attn", "linear_attn")
PROBE_VERSION = "attn_probe/2026-11-03.1"

_S: dict[str, Any] = dict(
    installed=False, armed=False, tag=None, records=[], layers=[], model_ev=None,
    logits_ev=None, layer_ev=[], layer_tokens=[], step_meta=None, step_idx=0, errors=[],
    model_cls=None, logits_calls=0,
)


def _ev():
    return torch.cuda.Event(enable_timing=True, external=True)


def _first_tokens(args, kwargs):
    for v in list(args) + list(kwargs.values()):
        if isinstance(v, torch.Tensor) and v.dim() >= 2:
            return int(v.shape[0])
    return None


def _wrap_forward(mod, start, end, idx=None):
    orig = mod.forward

    def fwd(*args, **kwargs):
        start.record()
        if idx is not None:
            try:
                _S["layer_tokens"][idx] = _first_tokens(args, kwargs)
            except Exception:
                pass
        out = orig(*args, **kwargs)
        end.record()
        return out

    mod.forward = fwd


def _instrument(model) -> dict:
    """Wrap model window, logits and per-layer attention modules (instance level)."""
    info = dict(model_cls=type(model).__name__, layers=[], probe_layers=PROBE_LAYERS)
    _S["model_cls"] = type(model).__name__
    ms, me = _ev(), _ev()
    _wrap_forward(model, ms, me)
    _S["model_ev"] = (ms, me)
    if hasattr(model, "compute_logits"):
        ls, le = _ev(), _ev()
        orig = model.compute_logits

        def cl(*a, **k):
            ls.record()
            out = orig(*a, **k)
            le.record()
            _S["logits_calls"] += 1
            return out

        model.compute_logits = cl
        _S["logits_ev"] = (ls, le)
    if PROBE_LAYERS:
        # Find decoder layers: the longest ModuleList whose items have an attention child.
        best = None
        for name, m in model.named_modules():
            if isinstance(m, torch.nn.ModuleList) and len(m) > 0:
                hits = [any(hasattr(x, c) and isinstance(getattr(x, c), torch.nn.Module)
                            for c in ATTN_CHILD_NAMES) for x in m]
                if sum(hits) and (best is None or len(m) > len(best[1])):
                    best = (name, m)
        if best is not None:
            lname, layers = best
            for i, layer in enumerate(layers):
                child = None
                for c in ATTN_CHILD_NAMES:
                    sub = getattr(layer, c, None)
                    if isinstance(sub, torch.nn.Module):
                        child = (c, sub)
                        break
                if child is None:
                    continue
                s, e = _ev(), _ev()
                idx = len(_S["layer_ev"])
                _S["layer_ev"].append((s, e))
                _S["layer_tokens"].append(None)
                _wrap_forward(child[1], s, e, idx)
                info["layers"].append(dict(idx=idx, layer=i, path=f"{lname}.{i}.{child[0]}",
                                           cls=type(child[1]).__name__))
    _S["layers"] = info["layers"]
    _S["installed"] = True
    return info


def _read_step(meta: dict) -> dict:
    torch.cuda.current_stream().synchronize()
    rec = dict(meta)
    rec["valid"] = True
    rec["why"] = []
    ms, me = _S["model_ev"]
    try:
        model_ms = ms.elapsed_time(me)
    except Exception as e:  # never recorded / not ready
        rec.update(valid=False, why=[f"model window: {e}"], model_ms=None)
        return rec
    rec["model_ms"] = model_ms
    if model_ms <= 0:
        rec["valid"] = False
        rec["why"].append("non-positive model window")
    if _S["logits_ev"] is not None and rec.get("logits_expected", True):
        ls, le = _S["logits_ev"]
        try:
            lo = ms.elapsed_time(ls)
            rec["logits_ms"] = ls.elapsed_time(le)
            if lo < model_ms - 1e-3:  # logits must follow this step's forward
                rec["logits_ms"] = None
                rec["why"].append("stale logits window")
        except Exception as e:
            rec["logits_ms"] = None
            rec["why"].append(f"logits: {e}")
    if PROBE_LAYERS:
        starts, durs, toks = [], [], []
        bad = 0
        for i, (s, e) in enumerate(_S["layer_ev"]):
            try:
                off = ms.elapsed_time(s)
                end_off = ms.elapsed_time(e)
                d = s.elapsed_time(e)
            except Exception:
                bad += 1
                starts.append(None)
                durs.append(None)
                continue
            if off < -1e-3 or end_off > model_ms + 1e-3 or d < 0:
                bad += 1
            starts.append(round(off, 5))
            durs.append(round(d, 5))
        rec["layer_start_ms"] = starts
        rec["layer_ms"] = durs
        rec["layer_tokens"] = list(_S["layer_tokens"]) if meta.get("eager_tokens") else None
        rec["regions_valid"] = len(durs) - bad
        rec["regions_total"] = len(durs)
        if bad:
            rec["valid"] = False
            rec["why"].append(f"{bad} stale/out-of-window regions")
        rec["attn_ms"] = sum(d for d in durs if d is not None) if not bad else None
    return rec


def _sched_meta(so) -> dict:
    m = dict(total_tokens=int(so.total_num_scheduled_tokens),
             num_reqs=len(so.num_scheduled_tokens))
    reqs = {}
    try:
        for r in so.scheduled_new_reqs:
            reqs[r.req_id] = dict(computed=int(r.num_computed_tokens), new=True)
        c = so.scheduled_cached_reqs
        for i, rid in enumerate(c.req_ids):
            reqs[rid] = dict(computed=int(c.num_computed_tokens[i]), new=False)
        for rid, n in so.num_scheduled_tokens.items():
            reqs.setdefault(rid, {})["scheduled"] = int(n)
    except Exception as e:
        m["meta_error"] = repr(e)
    m["reqs"] = reqs
    return m


def _patch_worker():
    from vllm.v1.worker.gpu_worker import Worker

    if getattr(Worker, "_attn_probe_patched", False):
        return
    Worker._attn_probe_patched = True
    orig_load, orig_exec, orig_sample = Worker.load_model, Worker.execute_model, Worker.sample_tokens

    def load_model(self, *a, **k):
        out = orig_load(self, *a, **k)
        try:
            self._attn_probe_info = _instrument(self.model_runner.get_model())
        except Exception:
            self._attn_probe_info = dict(error=traceback.format_exc())
        return out

    def execute_model(self, scheduler_output, *a, **k):
        meta = None
        if _S["armed"] and _S["installed"] and scheduler_output.total_num_scheduled_tokens > 0:
            meta = _sched_meta(scheduler_output)
            meta.update(step=_S["step_idx"], tag=_S["tag"], t_host=time.time())
            meta["eager_tokens"] = True
            _S["step_idx"] += 1
        _S["step_meta"] = meta
        # Host-side token counts are only refreshed when Python module code runs (eager);
        # under graph replay they stay None.
        _S["layer_tokens"] = [None] * len(_S["layer_ev"])
        out = orig_exec(self, scheduler_output, *a, **k)
        if meta is not None and out is not None:
            meta["logits_expected"] = False
            try:
                _S["records"].append(_read_step(meta))
            except Exception:
                _S["errors"].append(traceback.format_exc())
            _S["step_meta"] = None
        return out

    def sample_tokens(self, *a, **k):
        out = orig_sample(self, *a, **k)
        meta = _S["step_meta"]
        if meta is not None:
            try:
                _S["records"].append(_read_step(meta))
            except Exception:
                _S["errors"].append(traceback.format_exc())
            _S["step_meta"] = None
        return out

    Worker.load_model = load_model
    Worker.execute_model = execute_model
    Worker.sample_tokens = sample_tokens


def _jsonable(x, depth=0):
    if depth > 6:
        return repr(x)
    if x is None or isinstance(x, (bool, int, float, str)):
        return x
    if isinstance(x, torch.dtype):
        return str(x)
    if isinstance(x, (list, tuple)):
        return [_jsonable(v, depth + 1) for v in x]
    if isinstance(x, dict):
        return {str(k): _jsonable(v, depth + 1) for k, v in x.items()}
    if hasattr(x, "__dataclass_fields__"):
        d = {"__cls__": type(x).__name__}
        for f in x.__dataclass_fields__:
            try:
                d[f] = _jsonable(getattr(x, f), depth + 1)
            except Exception as e:
                d[f] = f"<error {type(e).__name__}>"
        for p in ("page_size_bytes", "real_page_size_bytes", "type_id"):
            try:
                d[p] = _jsonable(getattr(x, p), depth + 1)
            except Exception as e:
                d[p] = f"<error {type(e).__name__}>"
        return d
    if hasattr(x, "value") and hasattr(x, "name"):  # enums
        return str(x.name)
    return repr(x)


def _tensor_desc(t: torch.Tensor) -> dict:
    try:
        st = t.untyped_storage()
        return dict(shape=list(t.shape), dtype=str(t.dtype), stride=list(t.stride()),
                    data_ptr=int(t.data_ptr()), storage_ptr=int(st.data_ptr()),
                    storage_nbytes=int(st.nbytes()), nbytes=int(t.numel() * t.element_size()),
                    device=str(t.device))
    except Exception as e:
        return dict(error=repr(e))


def _collect_tensors(x, out, depth=0):
    if depth > 3:
        return
    if isinstance(x, torch.Tensor):
        out.append(_tensor_desc(x))
    elif isinstance(x, (list, tuple)):
        for v in x:
            _collect_tensors(v, out, depth + 1)
    elif isinstance(x, dict):
        for v in x.values():
            _collect_tensors(v, out, depth + 1)


class AttnProbeExt:
    """Methods callable through ``LLM.collective_rpc``; all return data, never raise."""

    def probe_info(self) -> dict:
        try:
            import torch.distributed as dist

            rank = dist.get_rank() if dist.is_initialized() else 0
            moe = None
            try:
                for name, m in self.model_runner.get_model().named_modules():
                    if hasattr(m, "local_num_experts") and hasattr(m, "global_num_experts"):
                        moe = dict(module=name, cls=type(m).__name__)
                        for a in ("local_num_experts", "global_num_experts", "ep_size", "ep_rank",
                                  "tp_size", "tp_rank", "use_ep", "dp_size"):
                            if hasattr(m, a):
                                moe[a] = _jsonable(getattr(m, a))
                        for a in ("use_ep", "ep_size", "tp_size", "ep_rank"):
                            pc = getattr(m, "moe_parallel_config", None)
                            if pc is not None and hasattr(pc, a):
                                moe["parallel_" + a] = _jsonable(getattr(pc, a))
                        break
            except Exception as e:
                moe = dict(error=repr(e))
            return dict(rank=rank, version=PROBE_VERSION, probe_layers=PROBE_LAYERS, moe=moe,
                        info=getattr(self, "_attn_probe_info", None),
                        device=torch.cuda.current_device(),
                        uuid=str(torch.cuda.get_device_properties(torch.cuda.current_device()).uuid),
                        runner=type(self.model_runner).__module__ + "." + type(self.model_runner).__name__,
                        errors=_S["errors"][-5:])
        except Exception:
            return dict(error=traceback.format_exc())

    def probe_arm(self, tag: str | None) -> dict:
        _S["armed"] = tag is not None
        _S["tag"] = tag
        return dict(ok=True)

    def probe_fetch(self, clear: bool = True) -> dict:
        try:
            import torch.distributed as dist

            rank = dist.get_rank() if dist.is_initialized() else 0
            recs = _S["records"]
            errs = _S["errors"]
            if clear:
                _S["records"] = []
                _S["errors"] = []
            if rank != 0:  # scheduler metadata is identical on all ranks; keep rank 0's copy
                recs = [{k: v for k, v in r.items() if k != "reqs"} for r in recs]
            return dict(rank=rank, records=recs, errors=errs)
        except Exception:
            return dict(error=traceback.format_exc())

    def probe_mem(self) -> dict:
        try:
            torch.cuda.synchronize()
            free, total = torch.cuda.mem_get_info()
            return dict(allocated=torch.cuda.memory_allocated(), reserved=torch.cuda.memory_reserved(),
                        free=free, total=total, t=time.time())
        except Exception:
            return dict(error=traceback.format_exc())

    def probe_state_buffers(self) -> dict:
        """Describe every attention/state layer's live cache tensors, dtypes and scales."""
        try:
            import torch.distributed as dist

            rank = dist.get_rank() if dist.is_initialized() else 0
            ctx = self.vllm_config.compilation_config.static_forward_context
            layers = {}
            for name, layer in ctx.items():
                d = dict(cls=type(layer).__module__ + "." + type(layer).__name__)
                tens = []
                for attr in ("kv_cache", "_kv_cache", "state", "cache"):
                    if hasattr(layer, attr):
                        try:
                            _collect_tensors(getattr(layer, attr), tens)
                        except Exception:
                            pass
                d["tensors"] = tens
                for attr in ("kv_cache_dtype", "cache_dtype", "kv_sharing_target_layer_name",
                             "sliding_window", "num_kv_heads", "head_size", "head_size_v",
                             "dtype", "attn_backend"):
                    if hasattr(layer, attr):
                        try:
                            v = getattr(layer, attr)
                            d[attr] = v.get_name() if hasattr(v, "get_name") else _jsonable(v)
                        except Exception as e:
                            d[attr] = f"<error {type(e).__name__}>"
                for attr in ("_k_scale_float", "_v_scale_float", "_q_scale_float"):
                    if hasattr(layer, attr):
                        try:
                            d[attr] = float(getattr(layer, attr))
                        except Exception:
                            pass
                try:
                    if hasattr(layer, "get_kv_cache_spec"):
                        d["spec"] = _jsonable(layer.get_kv_cache_spec(self.vllm_config))
                except Exception as e:
                    d["spec"] = f"<error {type(e).__name__}: {e}>"
                layers[name] = d
            kvc = getattr(self.model_runner, "kv_cache_config", None)
            raw = []
            kv = getattr(self.model_runner, "kv_caches", None)
            if kv is not None:
                _collect_tensors(kv, raw)
            return dict(rank=rank, layers=layers, kv_cache_config=_jsonable(kvc),
                        runner_kv_caches=raw)
        except Exception:
            return dict(error=traceback.format_exc())


_patch_worker()
