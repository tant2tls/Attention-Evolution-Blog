"""Own one model launch: process group, timeout, cleanup, shutdown verification, completion row.

Usage: python bench/session.py --model-key K --session-id ID [--expected-points N] [--timeout S]
       [--no-completion-row] -- <child command ...>

* Refuses to launch if the session id was used before or any GPU holds > 2 GiB (unowned work).
* Starts the child in its own session/process group, tees output to logs/<model>__<id>.log.
* On exit, timeout, SIGINT or SIGTERM: SIGTERM then SIGKILL to the group, reap the leader
  (experience: zombie leader kept killpg(pgid,0) true), wait until no process of the group remains
  and every GPU's used memory is back within 1 GiB of the recorded idle baseline.
* Appends one row to model_completion.csv (never rewrites earlier rows).
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import study_config as sc  # noqa: E402

FIELDS = ["study_id", "model", "revision", "session_id", "started_at_utc", "measurement_completed_at_utc",
          "shutdown_verified_at_utc", "finished_at_utc", "elapsed_seconds", "valid_points", "expected_points",
          "valid_repetitions", "expected_repetitions", "status", "reason", "manifest", "log"]


def utc() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def gpu_used_mib() -> list[int]:
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, timeout=60).stdout
    return [int(x) for x in out.split()]


def group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-key", required=True)
    ap.add_argument("--session-id", required=True)
    ap.add_argument("--study-dir", default=str(sc.STUDY_DIR))
    ap.add_argument("--expected-points", type=int, default=0)
    ap.add_argument("--timeout", type=float, default=5400)
    ap.add_argument("--no-completion-row", action="store_true")
    ap.add_argument("--role", default="")
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
    study = Path(a.study_dir)
    logs = study / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    life = logs / f"{a.model_key}__{a.session_id}__lifecycle.json"
    logf = logs / f"{a.model_key}__{a.session_id}.log"
    if life.exists() or logf.exists():
        print(f"session id {a.session_id} already used; refusing", file=sys.stderr)
        return 10
    used = gpu_used_mib()
    if max(used) > 2048:
        print(f"GPUs busy {used}; refusing to launch over other work", file=sys.stderr)
        life.write_text(json.dumps(dict(status="refused", gpu_used_mib=used, t=utc())))
        return 11
    baseline = used
    rec = dict(study_id=study.name, model=a.model_key, session_id=a.session_id, role=a.role, cmd=cmd,
               idle_baseline_mib=baseline, started_at_utc=utc(), host_cpu_quota=Path("/sys/fs/cgroup/cpu.max").read_text().strip()
               if Path("/sys/fs/cgroup/cpu.max").exists() else None)
    t0 = time.monotonic()
    life.write_text(json.dumps(rec, indent=1))
    measurement_completed = None
    interrupted = {"sig": None}

    def on_sig(signum, _frame):
        interrupted["sig"] = signum

    signal.signal(signal.SIGINT, on_sig)
    signal.signal(signal.SIGTERM, on_sig)
    with open(logf, "wb") as lf:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True,
                                cwd=str(sc.REPO))
        pgid = os.getpgid(proc.pid)
        rec["pgid"] = pgid
        life.write_text(json.dumps(rec, indent=1))
        os.set_blocking(proc.stdout.fileno(), False)
        timed_out = False
        buf = b""
        while True:
            chunk = None
            try:
                chunk = proc.stdout.read()
            except Exception:
                pass
            if chunk:
                lf.write(chunk)
                lf.flush()
                buf = (buf + chunk)[-20000:]
                if b"MEASUREMENT_COMPLETED" in buf and measurement_completed is None:
                    measurement_completed = utc()
            if proc.poll() is not None:
                rest = proc.stdout.read() or b""
                lf.write(rest)
                if b"MEASUREMENT_COMPLETED" in rest and measurement_completed is None:
                    measurement_completed = utc()
                break
            if interrupted["sig"] is not None:
                break
            if time.monotonic() - t0 > a.timeout:
                timed_out = True
                break
            time.sleep(0.5)
        rc = proc.poll()
        # ---- cleanup of the owned group
        for sig, wait in ((signal.SIGTERM, 30), (signal.SIGKILL, 60)):
            if not group_alive(pgid):
                break
            try:
                os.killpg(pgid, sig)
            except ProcessLookupError:
                break
            end = time.monotonic() + wait
            while time.monotonic() < end:
                proc.poll()  # reap the leader so a zombie does not keep the group "alive"
                if not group_alive(pgid):
                    break
                time.sleep(1)
        proc.poll()
        rc = proc.returncode if rc is None else rc
    # ---- verify shutdown: group gone and GPU memory back to baseline
    shutdown_ok = False
    end = time.monotonic() + 300
    while time.monotonic() < end:
        now = gpu_used_mib()
        if not group_alive(pgid) and all(n <= b + 1024 for n, b in zip(now, baseline)):
            shutdown_ok = True
            break
        time.sleep(2)
    shutdown_at = utc() if shutdown_ok else ""
    finished = utc()
    elapsed = round(time.monotonic() - t0, 1)
    if interrupted["sig"] is not None:
        status, reason = "interrupted", f"signal {interrupted['sig']}"
    elif timed_out:
        status, reason = "interrupted", f"timeout {a.timeout}s"
    elif rc == 0:
        status, reason = "complete", ""
    else:
        status, reason = "failed", f"exit code {rc}"
    if not shutdown_ok:
        status, reason = ("failed" if status == "complete" else status), (reason + "; shutdown not verified").strip("; ")
    rec.update(returncode=rc, timed_out=timed_out, measurement_completed_at_utc=measurement_completed,
               shutdown_verified_at_utc=shutdown_at, finished_at_utc=finished, elapsed_seconds=elapsed,
               gpu_after_mib=gpu_used_mib(), status=status, reason=reason)
    life.write_text(json.dumps(rec, indent=1))
    if not a.no_completion_row:
        valid_points = valid_reps = 0
        sess = study / "raw" / a.model_key / "sessions" / a.session_id / "session_summary.json"
        if sess.exists():
            pts = json.loads(sess.read_text()).get("points", [])
            if pts and isinstance(pts[0], list):
                ok = {}
                for name, rep, st, *_ in pts:
                    ok.setdefault(name, 0)
                    ok[name] += st == "measured"
                valid_reps = sum(ok.values())
                valid_points = sum(1 for v in ok.values() if v >= sc.REPEATS)
            else:
                valid_points = sum(1 for p in pts if p.get("status") == "measured")
                valid_reps = valid_points
        if status == "complete" and a.expected_points and valid_points < a.expected_points:
            status, reason = "partial", f"{valid_points}/{a.expected_points} points with 3 valid repetitions"
        row = dict(study_id=study.name, model=a.model_key, revision=sc.MODELS[a.model_key]["revision"],
                   session_id=a.session_id, started_at_utc=rec["started_at_utc"],
                   measurement_completed_at_utc=measurement_completed or "", shutdown_verified_at_utc=shutdown_at,
                   finished_at_utc=finished, elapsed_seconds=elapsed, valid_points=valid_points,
                   expected_points=a.expected_points, valid_repetitions=valid_reps,
                   expected_repetitions=a.expected_points * (sc.REPEATS if a.expected_points else 0),
                   status=status, reason=(a.role + ": " if a.role else "") + reason,
                   manifest=f"raw/{a.model_key}/sessions/{a.session_id}/",
                   log=f"logs/{logf.name}")
        mc = study / "model_completion.csv"
        new = not mc.exists()
        with open(mc, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            if new:
                w.writeheader()
            w.writerow(row)
    print(json.dumps({k: rec[k] for k in ("status", "reason", "returncode", "elapsed_seconds",
                                          "shutdown_verified_at_utc")}), flush=True)
    return 0 if status == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
