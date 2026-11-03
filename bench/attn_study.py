"""Memory / decode / prefill attention runner for one model launch (one vLLM engine).

Modes
  smoke     validation launch: short warmup + 1K b1, 1K b8, 16K prefill, max-context b8 decode.
  study     accepted session: warmup, then 3 rep-major repetitions of every supported point
            (decode b1/2/4/8 + co-collected memory at b1, prefill) for 1K/16K/64K/128K.
  overhead  matched control with ATTN_PROBE_LAYERS=0 (model/logits windows only), same grid.

Runs the engine core in-process (VLLM_ENABLE_V1_MULTIPROCESSING=0) so the scheduler's live block
tables are readable between steps; TP8 workers run in separate processes with the attn_probe
worker extension. Every point is written to disk immediately after it finishes.
"""
from __future__ import annotations

import argparse
import datetime as dt
import gzip
import json
import math
import os
import platform
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.environ["PYTHONPATH"] = str(HERE) + (":" + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else "")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

import study_config as sc  # noqa: E402
import workload as wl  # noqa: E402


# Deterministic per-label prompt salt (Python's str hash is randomized per process).
LABEL_SALT = {"accepted": 0, "warmup": 1, "smoke": 2}


def utc() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def wjson(path: Path, obj, gz: bool = False):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():  # never overwrite evidence (experience F5)
        raise FileExistsError(path)
    data = json.dumps(obj, indent=None, default=str).encode()
    if gz:
        with gzip.open(str(path) + ".gz" if not str(path).endswith(".gz") else path, "wb") as f:
            f.write(data)
    else:
        path.write_bytes(data)


class Runner:
    def __init__(self, a):
        self.a = a
        self.key = a.model_key
        self.m = sc.MODELS[self.key]
        self.sid = a.session_id
        self.study = Path(a.study_dir)
        self.raw = self.study / "raw" / self.key
        self.sess_dir = self.raw / "sessions" / self.sid
        if self.sess_dir.exists():
            raise SystemExit(f"session dir exists, refusing to reuse id: {self.sess_dir}")
        self.sess_dir.mkdir(parents=True)
        self.probe_layers = os.environ.get("ATTN_PROBE_LAYERS", "1") != "0"
        self.events = []

    # ------------------------------------------------------------------ engine
    def launch(self):
        from vllm import LLM

        kw = sc.engine_kwargs(self.key, self.a.kv_cache_dtype)
        kw["worker_extension_cls"] = "attn_probe.AttnProbeExt"
        kw["disable_log_stats"] = False
        self.engine_kwargs = kw
        cfg_path = self.study / "config" / f"{self.key}__{self.sid}__launch.json"
        wjson(cfg_path, dict(engine_kwargs=kw, env={k: v for k, v in os.environ.items()
                                                   if k.startswith(("VLLM_", "ATTN_", "CUDA_", "NCCL_", "TORCH", "HF_HUB"))},
                             argv=sys.argv, started_at_utc=utc()))
        t0 = time.monotonic()
        self.llm = LLM(**kw)
        self.init_s = time.monotonic() - t0
        self.engine = self.llm.llm_engine
        self.core = self.engine.engine_core.engine_core
        self.sched = self.core.scheduler
        self.tok = self.llm.get_tokenizer()
        self.stream = wl.token_stream(self.key, self.tok)
        self.record_resolved()

    def internal_id(self, rid: str) -> str:
        """vLLM appends a random suffix to request ids (assign_request_id); find the scheduler's id."""
        hits = [k for k in self.sched.requests if k == rid or k.startswith(rid + "-")]
        if len(hits) != 1:
            raise RuntimeError(f"cannot resolve internal id for {rid}: {hits}")
        return hits[0]

    def rpc(self, name, *args):
        return self.llm.collective_rpc(name, args=args)

    def record_resolved(self):
        import torch
        import vllm

        vc = self.engine.vllm_config
        pc, cc, mc, sch, comp = (vc.parallel_config, vc.cache_config, vc.model_config,
                                 vc.scheduler_config, vc.compilation_config)
        g = lambda o, n: (lambda v: v if isinstance(v, (int, float, str, bool, type(None))) else str(v))(getattr(o, n, None))
        resolved = dict(
            vllm=vllm.__version__, torch=torch.__version__, torch_cuda=torch.version.cuda,
            python=platform.python_version(),
            tp=g(pc, "tensor_parallel_size"), ep=g(pc, "enable_expert_parallel"),
            dp=g(pc, "data_parallel_size"), pp=g(pc, "pipeline_parallel_size"),
            gpu_memory_utilization=g(cc, "gpu_memory_utilization"),
            kv_cache_dtype=g(cc, "cache_dtype"), block_size=g(cc, "block_size"),
            prefix_caching=g(cc, "enable_prefix_caching"),
            num_gpu_blocks=g(cc, "num_gpu_blocks"),
            max_model_len=g(mc, "max_model_len"), model_dtype=g(mc, "dtype"),
            quantization=g(mc, "quantization"), architectures=str(getattr(mc, "architectures", None)),
            hf_config_cls=type(mc.hf_config).__name__,
            max_num_batched_tokens=g(sch, "max_num_batched_tokens"), max_num_seqs=g(sch, "max_num_seqs"),
            async_scheduling=g(sch, "async_scheduling"), chunked_prefill=g(sch, "enable_chunked_prefill"),
            speculative_config=str(vc.speculative_config),
            cudagraph_mode=g(comp, "cudagraph_mode"), compilation_mode=g(comp, "mode"),
            cudagraph_capture_sizes=str(getattr(comp, "cudagraph_capture_sizes", None)),
            breakable_cudagraph=os.environ.get("VLLM_USE_BREAKABLE_CUDAGRAPH"),
            use_v2_model_runner=str(getattr(vc, "use_v2_model_runner", None)),
            init_seconds=round(self.init_s, 1),
        )
        self.resolved = resolved
        info = self.rpc("probe_info")
        self.probe_info = info
        bufs = self.rpc("probe_state_buffers")
        try:
            sched_kvc = __import__("attn_probe")._jsonable(self.sched.kv_cache_config)
        except Exception:
            sched_kvc = traceback.format_exc()
        wjson(self.sess_dir / "resolved_config.json", resolved)
        wjson(self.sess_dir / "probe_info.json", info)
        wjson(self.sess_dir / "state_buffers.json", bufs, gz=True)
        wjson(self.sess_dir / "scheduler_kv_cache_config.json", sched_kvc)
        # ---- policy gate: refuse to measure if the resolved config deviates
        errs = []
        if resolved["tp"] != 8:
            errs.append("tp != 8")
        if resolved["ep"] is not (not self.m["dense"]):
            errs.append("ep setting differs from policy (on for MoE, N/A=off for dense)")
        if abs(float(resolved["gpu_memory_utilization"]) - 0.82) > 1e-9:
            errs.append("utilization != 0.82")
        if resolved["prefix_caching"]:
            errs.append("prefix caching on")
        if resolved["max_num_batched_tokens"] != 8192 or resolved["max_num_seqs"] != 8:
            errs.append("scheduler budget mismatch")
        if resolved["async_scheduling"]:
            errs.append("async scheduling on")
        if resolved["speculative_config"] not in ("None", None):
            errs.append("speculation configured")
        if resolved["vllm"] != sc.VLLM_VERSION or resolved["torch"] != sc.TORCH_VERSION:
            errs.append("runtime version mismatch")
        if any("error" in (r or {}) for r in info) or any((r.get("info") or {}).get("error") for r in info):
            errs.append("probe install error")
        if self.probe_layers:
            nreg = {len((r.get("info") or {}).get("layers", [])) for r in info}
            if nreg != {self.m["expected_regions"]}:
                errs.append(f"attention regions {nreg} != expected {self.m['expected_regions']}")
        self.policy_errors = errs
        print("RESOLVED", json.dumps(resolved), flush=True)
        print("POLICY_ERRORS", errs, flush=True)
        print("PROBE_INFO_RANK0", json.dumps(info[0])[:3000], flush=True)

    # --------------------------------------------------------------- helpers
    def snapshot(self, rids, phase):
        snap = dict(phase=phase, t=utc(), reqs={})
        try:
            kvm = self.sched.kv_cache_manager
            for rid in rids:
                req = self.sched.requests.get(rid)
                if req is None:
                    continue
                groups = []
                for blist in kvm.get_blocks(rid).blocks:
                    ids = [int(b.block_id) for b in blist]
                    live = [int(b.block_id) for b in blist if not b.is_null]
                    groups.append(dict(n_blocks=len(ids), n_live=len(live), live_ids=live))
                snap["reqs"][rid] = dict(num_computed_tokens=int(req.num_computed_tokens),
                                         num_prompt_tokens=int(req.num_prompt_tokens),
                                         num_output_tokens=len(req.output_token_ids), groups=groups)
            bp = kvm.block_pool
            snap["pool"] = dict(num_gpu_blocks=int(bp.num_gpu_blocks), free=int(bp.get_num_free_blocks()))
        except Exception:
            snap["error"] = traceback.format_exc()
        snap["mem"] = self.rpc("probe_mem")
        return snap

    def drive(self, rids, prompt_len, batch, snapshots: bool, settle_steps: int, max_steps: int):
        """Step the engine until all requests finish; optional memory snapshots."""
        snaps = []
        n = 0
        decode_steps_after_prefill = 0
        took_post, took_settled = False, False
        prefill_done_step = None
        while self.engine.has_unfinished_requests():
            self.engine.step()
            n += 1
            if n > max_steps:
                raise RuntimeError(f"exceeded {max_steps} engine steps")
            live = [self.sched.requests.get(r) for r in rids]
            live = [r for r in live if r is not None]
            all_pref = len(live) == batch and all(r.num_computed_tokens >= prompt_len for r in live)
            if all_pref and prefill_done_step is None:
                prefill_done_step = n
                if snapshots and not took_post:
                    snaps.append(self.snapshot(rids, "post_prefill"))
                    took_post = True
            elif prefill_done_step is not None and len(live) == batch:
                decode_steps_after_prefill += 1
                if snapshots and not took_settled and decode_steps_after_prefill >= settle_steps:
                    snaps.append(self.snapshot(rids, "settled_decode"))
                    took_settled = True
        return snaps, n

    def fetch(self):
        out = self.rpc("probe_fetch", True)
        errs = [e for r in out for e in r.get("errors", [])]
        return [r.get("records", []) for r in out], errs

    @staticmethod
    def join_ranks(per_rank):
        """Return list of steps: rank-0 meta plus per-rank timing lists (index aligned by step id)."""
        by_step = {}
        for rank, recs in enumerate(per_rank):
            for r in recs:
                by_step.setdefault(r["step"], {})[rank] = r
        steps = []
        for s in sorted(by_step):
            ranks = by_step[s]
            r0 = ranks.get(0, {})
            st = dict(step=s, total_tokens=r0.get("total_tokens"), num_reqs=r0.get("num_reqs"),
                      reqs=r0.get("reqs"), n_ranks=len(ranks),
                      valid=len(ranks) == len(per_rank) and all(x.get("valid") for x in ranks.values()),
                      why=sorted({w for x in ranks.values() for w in x.get("why", [])}))
            st["rank_model_ms"] = [ranks[k].get("model_ms") for k in sorted(ranks)]
            st["rank_logits_ms"] = [ranks[k].get("logits_ms") for k in sorted(ranks)]
            st["rank_attn_ms"] = [ranks[k].get("attn_ms") for k in sorted(ranks)]
            st["rank0_layer_ms"] = r0.get("layer_ms")
            st["rank0_layer_start_ms"] = r0.get("layer_start_ms")
            st["rank0_layer_tokens"] = r0.get("layer_tokens")
            st["rank_layer_ms"] = [ranks[k].get("layer_ms") for k in sorted(ranks)]
            st["regions"] = [ranks[k].get("regions_valid") for k in sorted(ranks)]
            def node(vals):
                return None if any(v is None for v in vals) or not vals else max(vals)
            st["node_attn_ms"] = node(st["rank_attn_ms"]) if st["valid"] else None
            whole = [None if (m is None or l is None) else m + l
                     for m, l in zip(st["rank_model_ms"], st["rank_logits_ms"])]
            st["node_forward_ms"] = node(st["rank_model_ms"])
            st["node_model_ms"] = node(whole)
            steps.append(st)
        return steps

    def point_path(self, exp, rep, point, ok):
        base = self.raw / exp / (f"repetition-{rep}" if ok else "failed-attempts")
        name = f"{self.sid}__{point}" + ("" if ok else f"__rep{rep}") + ".json"
        p = base / name
        k = 1
        while p.exists() or Path(str(p) + ".gz").exists():
            p = base / "collisions" / f"{name}.{k}"
            k += 1
        return p

    def common(self):
        return dict(model=self.key, revision=self.m["revision"], session=self.sid,
                    kv_cache_dtype_requested=self.a.kv_cache_dtype, probe_layers=self.probe_layers,
                    probe_version=__import__("attn_probe").PROBE_VERSION, workload=wl.describe())

    # ------------------------------------------------------------ experiments
    def run_decode(self, L, B, rep, label, memory: bool):
        point = f"decode_L{L}_B{B}"
        res = dict(self.common(), experiment="decode", point=point, context_tokens=L, target_batch=B,
                   repeat=rep, label=label, started_at_utc=utc())
        n_pref_steps = math.ceil(B * L / sc.COMMON["max_num_batched_tokens"])
        max_tokens = n_pref_steps + sc.DECODE_WARMUP_STEPS + sc.DECODE_MEASURED_STEPS + 8
        res["max_tokens"] = max_tokens
        from vllm import SamplingParams

        sp = SamplingParams(max_tokens=max_tokens, min_tokens=max_tokens, ignore_eos=True,
                            temperature=0.0, detokenize=False)
        rids, iids, hashes = [], [], []
        try:
            res["mem_baseline"] = self.rpc("probe_mem") if memory else None
            self.rpc("probe_arm", f"{label}:{point}:r{rep}")
            for i in range(B):
                ids = wl.prompt(self.stream, L, i, rep, salt=LABEL_SALT.get(label, 5))
                rid = f"{self.sid}-{label}-{point}-r{rep}-{i}"
                self.engine.add_request(rid, {"prompt_token_ids": ids}, sp)
                rids.append(rid)
                iids.append(self.internal_id(rid))
                hashes.append(wl.prompt_hash(ids))
            res["prompt_hashes"] = hashes
            res["prompt_tokens"] = [L] * B
            res["internal_request_ids"] = iids
            t0 = time.monotonic()
            snaps, nsteps = self.drive(iids, L, B, memory, sc.DECODE_WARMUP_STEPS, 100000)
            res["wall_s"] = round(time.monotonic() - t0, 3)
            self.rpc("probe_arm", None)
            per_rank, errs = self.fetch()
            res["probe_errors"] = errs
            steps = self.join_ranks(per_rank)
            res["engine_steps"] = nsteps
            res["snapshots"] = snaps
            # pure decode: exactly B requests, each scheduled 1 token, all past the prompt
            def pure(st):
                rq = st.get("reqs") or {}
                return (st["num_reqs"] == B and len(rq) == B and
                        all(v.get("scheduled") == 1 and v.get("computed", -1) >= L for v in rq.values()))
            idx = [i for i, st in enumerate(steps) if pure(st)]
            last_mixed = max([i for i, st in enumerate(steps) if not pure(st) and
                              any(v.get("computed", 0) < L for v in (st.get("reqs") or {}).values())] or [-1])
            run = [i for i in idx if i > last_mixed]
            need = sc.DECODE_WARMUP_STEPS + sc.DECODE_MEASURED_STEPS
            status, why = "measured", []
            consecutive = len(run) >= need and run[:need] == list(range(run[0], run[0] + need))
            if not consecutive:
                status, why = "failed", [f"only {len(run)} consecutive pure decode steps (need {need})"]
            warm = run[:sc.DECODE_WARMUP_STEPS]
            meas = run[sc.DECODE_WARMUP_STEPS:need]
            if status == "measured":
                bad = [steps[i]["step"] for i in meas if not steps[i]["valid"]]
                if bad:
                    status = "failed"
                    why.append(f"invalid probe steps {bad[:5]} ({len(bad)}): " +
                               "; ".join(sorted({w for i in meas for w in steps[i]["why"]}))[:500])
            res["warmup_step_ids"] = [steps[i]["step"] for i in warm]
            res["measured_step_ids"] = [steps[i]["step"] for i in meas]
            res["steps"] = steps
            if status == "measured":
                ms = [steps[i] for i in meas]
                ctx = [[v["computed"] + 1 for v in st["reqs"].values()] for st in ms]
                res.update(
                    actual_batch=sorted({st["num_reqs"] for st in ms}),
                    context_start_tokens=min(ctx[0]), context_end_tokens=max(ctx[-1]),
                    context_first_step=ctx[0], context_last_step=ctx[-1],
                    warmup_steps=len(warm), measured_steps=len(ms),
                    model_ms_per_step=sum(st["node_model_ms"] for st in ms) / len(ms),
                    forward_ms_per_step=sum(st["node_forward_ms"] for st in ms) / len(ms),
                    model_ms_samples=[st["node_model_ms"] for st in ms],
                )
                if self.probe_layers:
                    att = [st["node_attn_ms"] for st in ms]
                    res["attention_ms_per_step"] = sum(att) / len(att)
                    res["attention_ms_samples"] = att
                    res["effective_attention_tokens_per_s"] = B / (res["attention_ms_per_step"] / 1000.0)
                    res["rank_attention_ms_mean"] = [sum(st["rank_attn_ms"][k] for st in ms) / len(ms)
                                                     for k in range(len(ms[0]["rank_attn_ms"]))]
                if len(res["actual_batch"]) != 1 or res["actual_batch"][0] != B:
                    status, why = "failed", why + ["actual batch differs"]
            res["status"], res["why"] = status, why
        except Exception:
            self.rpc("probe_arm", None)
            res["status"] = "failed"
            res["why"] = [traceback.format_exc()]
            self._abort(rids)
        res["finished_at_utc"] = utc()
        ok = res["status"] == "measured"
        p = self.point_path("decode", rep, point, ok)
        wjson(p, res, gz=True)  # per-step x per-rank x per-layer records are large; gzip is lossless
        p = Path(str(p) + ".gz")
        print(f"POINT {point} rep{rep} {label} {res['status']} attn={res.get('attention_ms_per_step')} "
              f"model={res.get('model_ms_per_step')} ctx={res.get('context_start_tokens')}..{res.get('context_end_tokens')} "
              f"why={str(res.get('why'))[:300]}", flush=True)
        if memory:
            mres = self.write_memory(L, rep, label, res, p)
            self.last_memory = (mres["point"], rep, mres["status"])
        return res

    def write_memory(self, L, rep, label, dec, dec_path):
        point = f"memory_L{L}"
        snaps = dec.get("snapshots") or []
        ok = dec["status"] == "measured" and any(s["phase"] == "settled_decode" and "error" not in s for s in snaps)
        res = dict(self.common(), experiment="memory", point=point, context_tokens=L, repeat=rep, label=label,
                   co_collected_with=str(dec_path.relative_to(self.study)), snapshots=snaps,
                   mem_baseline=dec.get("mem_baseline"), status="measured" if ok else "failed",
                   why=[] if ok else ["decode run failed or settled snapshot missing"] + list(dec.get("why", [])))
        wjson(self.point_path("memory", rep, point, ok), res)
        return res

    def run_prefill(self, L, rep, label):
        chunk = sc.FIRST_CHUNK_1K if L == 1024 else sc.PREFILL_CHUNK
        point = f"prefill_L{L}"
        res = dict(self.common(), experiment="prefill", point=point, context_tokens=L, new_chunk_tokens=chunk,
                   repeat=rep, label=label, started_at_utc=utc())
        from vllm import SamplingParams

        sp = SamplingParams(max_tokens=1, ignore_eos=True, temperature=0.0, detokenize=False)
        rid = f"{self.sid}-{label}-{point}-r{rep}"
        try:
            ids = wl.prompt(self.stream, L, 7, rep, salt=3)
            res["prompt_hashes"] = [wl.prompt_hash(ids)]
            self.rpc("probe_arm", f"{label}:{point}:r{rep}")
            self.engine.add_request(rid, {"prompt_token_ids": ids}, sp)
            iid = self.internal_id(rid)
            res["internal_request_ids"] = [iid]
            t0 = time.monotonic()
            _, nsteps = self.drive([iid], L, 1, False, 0, 10000)
            res["wall_s"] = round(time.monotonic() - t0, 3)
            self.rpc("probe_arm", None)
            per_rank, errs = self.fetch()
            res["probe_errors"] = errs
            steps = self.join_ranks(per_rank)
            res["engine_steps"] = nsteps
            res["steps"] = steps
            sel = [st for st in steps if (st.get("reqs") or {}).get(iid, {}).get("computed") == L - chunk
                   and st["reqs"][iid].get("scheduled") == chunk and st["num_reqs"] == 1]
            status, why = "measured", []
            if len(sel) != 1:
                status, why = "failed", [f"final chunk step not found ({len(sel)})"]
            elif not sel[0]["valid"]:
                status, why = "failed", ["invalid probe step: " + "; ".join(sel[0]["why"])[:500]]
            if status == "measured":
                st = sel[0]
                res["measured_step_id"] = st["step"]
                res["model_ms_per_chunk"] = st["node_model_ms"]
                res["forward_ms_per_chunk"] = st["node_forward_ms"]
                res["chunk_start_tokens"] = L - chunk
                if self.probe_layers:
                    res["attention_ms_per_chunk"] = st["node_attn_ms"]
                    toks = st.get("rank0_layer_tokens") or []
                    hist = {}
                    for t in toks:
                        hist[str(t)] = hist.get(str(t), 0) + 1
                    res["executed_positions_by_layer"] = toks
                    res["executed_positions_hist"] = hist
                    if any(t is None for t in toks):
                        why.append("some layer token counts unavailable (not eager)")
            res["status"], res["why"] = status, why
        except Exception:
            self.rpc("probe_arm", None)
            res["status"], res["why"] = "failed", [traceback.format_exc()]
            self._abort([rid])
        res["finished_at_utc"] = utc()
        ok = res["status"] == "measured"
        wjson(self.point_path("prefill", rep, point, ok), res)
        print(f"POINT {point} rep{rep} {label} {res['status']} attn={res.get('attention_ms_per_chunk')} "
              f"model={res.get('model_ms_per_chunk')} hist={res.get('executed_positions_hist')} "
              f"why={str(res.get('why'))[:300]}", flush=True)
        return res

    def _abort(self, rids):
        try:
            self.engine.abort_request(rids)
            while self.engine.has_unfinished_requests():
                self.engine.step()
            self.fetch()
        except Exception:
            print("ABORT_FAILED", traceback.format_exc(), flush=True)

    # ----------------------------------------------------------------- plans
    def supported(self, L):
        mml = self.resolved["max_model_len"]
        need = L + math.ceil(8 * L / 8192) + sc.DECODE_WARMUP_STEPS + sc.DECODE_MEASURED_STEPS + 8
        return need <= mml, f"L + generated tokens ({need}) > max_model_len {mml}"

    def write_unsupported(self, L, why):
        for exp, pts in (("memory", [f"memory_L{L}"]), ("decode", [f"decode_L{L}_B{b}" for b in sc.DECODE_BATCHES]),
                         ("prefill", [f"prefill_L{L}"])):
            for pt in pts:
                p = self.raw / exp / "unsupported" / f"{self.sid}__{pt}.json"
                extra = dict(target_batch=int(pt.split("_B")[1])) if exp == "decode" else {}
                wjson(p, dict(self.common(), experiment=exp, point=pt, context_tokens=L, status="unsupported",
                              why=[why], t=utc(), **extra))

    def warmup(self):
        print("WARMUP start", utc(), flush=True)
        for B in sc.DECODE_BATCHES:
            self.run_decode(1024, B, 0, "warmup", False)
        self.run_prefill(16384, 0, "warmup")
        Lmax = max(L for L in sc.CONTEXTS if self.supported(L)[0])
        self.run_prefill(Lmax, 0, "warmup")
        print("WARMUP done", utc(), flush=True)

    def main(self):
        self.launch()
        if self.policy_errors:
            print("REFUSING TO MEASURE:", self.policy_errors, flush=True)
            return 3
        summary = dict(session=self.sid, mode=self.a.mode, points=[])
        if self.a.mode == "smoke":
            Lmax = max(L for L in sc.CONTEXTS if self.supported(L)[0])
            res = [self.run_decode(1024, 1, 0, "smoke", True), self.run_decode(1024, 8, 0, "smoke", False),
                   self.run_prefill(16384, 0, "smoke"), self.run_prefill(1024, 0, "smoke"),
                   self.run_decode(Lmax, 8, 0, "smoke", False)]
            summary["points"] = [dict(point=r["point"], status=r["status"], why=r.get("why"),
                                      attn=r.get("attention_ms_per_step", r.get("attention_ms_per_chunk")),
                                      model=r.get("model_ms_per_step", r.get("model_ms_per_chunk")))
                                 for r in res]
            summary["expected_regions"] = self.m["expected_regions"]
            summary["probe_layers"] = self.probe_layers
            summary["resolved"] = self.resolved
            summary["all_ok"] = all(r["status"] == "measured" for r in res)
        else:
            contexts = [int(x) for x in self.a.contexts.split(",")] if self.a.contexts else sc.CONTEXTS
            for L in contexts:
                ok, why = self.supported(L)
                if not ok:
                    self.write_unsupported(L, why)
            resume = [x for x in self.a.resume_sessions.split(",") if x]
            summary["resumed_from"] = resume

            def done(exp, point, rep):
                """Valid record of this (point, rep) from an earlier accepted session of the same config."""
                for ss in resume:
                    for ext in (".json", ".json.gz"):
                        if (self.raw / exp / f"repetition-{rep}" / f"{ss}__{point}{ext}").exists():
                            return ss
                return None

            self.warmup()
            for rep in range(1, sc.REPEATS + 1):
                for L in contexts:
                    if not self.supported(L)[0]:
                        continue
                    pre = done("prefill", f"prefill_L{L}", rep)
                    for B in sc.DECODE_BATCHES:
                        prev = done("decode", f"decode_L{L}_B{B}", rep)
                        if prev and (B != 1 or done("memory", f"memory_L{L}", rep)):
                            summary["points"].append([f"decode_L{L}_B{B}", rep, "measured", f"carried:{prev}"])
                            if B == 1:
                                summary["points"].append([f"memory_L{L}", rep, "measured", f"carried:{prev}"])
                            continue
                        self.last_memory = None
                        r = self.run_decode(L, B, rep, "accepted", memory=(B == 1 and self.probe_layers))
                        summary["points"].append([r["point"], rep, r["status"]])
                        if self.last_memory:  # co-collected memory point counts too (fixes.md B5)
                            summary["points"].append(list(self.last_memory))
                    if pre:
                        summary["points"].append([f"prefill_L{L}", rep, "measured", f"carried:{pre}"])
                        continue
                    r = self.run_prefill(L, rep, "accepted")
                    summary["points"].append([r["point"], rep, r["status"]])
        summary["measurement_completed_at_utc"] = utc()
        wjson(self.sess_dir / "session_summary.json", summary)
        print("SESSION_SUMMARY", json.dumps(summary, default=str)[:4000], flush=True)
        print("MEASUREMENT_COMPLETED", summary["measurement_completed_at_utc"], flush=True)
        if self.a.mode == "smoke":
            return 0 if summary["all_ok"] else 4
        return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-key", required=True)
    ap.add_argument("--study-dir", default=str(sc.STUDY_DIR))
    ap.add_argument("--session-id", required=True)
    ap.add_argument("--kv-cache-dtype", default="fp8")
    ap.add_argument("--mode", choices=["smoke", "study", "overhead"], default="study")
    ap.add_argument("--contexts", default="")
    ap.add_argument("--resume-sessions", default="", help="accepted sessions (same frozen config) whose valid "
                    "(point, rep) records are carried instead of re-measured")
    a = ap.parse_args()
    for k, v in sc.MODELS[a.model_key]["env"].items():  # model-specific runtime env, recorded in launch.json
        os.environ.setdefault(k, v)
    if a.mode == "overhead":
        os.environ["ATTN_PROBE_LAYERS"] = "0"
    r = Runner(a)
    rc = 1
    try:
        rc = r.main()
    except Exception:
        traceback.print_exc()
        rc = 2
    finally:
        try:
            r.llm.llm_engine.engine_core.shutdown()
        except Exception:
            pass
    sys.stdout.flush()
    os._exit(rc)


if __name__ == "__main__":
    main()
