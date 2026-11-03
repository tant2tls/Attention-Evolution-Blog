"""Download one pinned checkpoint revision into a study-owned directory and verify sizes.

Writes report/<study>/logs/download_<key>.json with start/end UTC, bytes, file list check and errors.
Never touches a shared HF pool: files go to WEIGHTS_ROOT/<key> via local_dir.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import study_config as sc  # noqa: E402


def utc() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-key", required=True)
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args()
    os.environ.pop("HF_HUB_OFFLINE", None)
    from fnmatch import fnmatch

    from huggingface_hub import HfApi, snapshot_download

    m = sc.MODELS[a.model_key]
    dest = sc.weights_dir(a.model_key)
    dest.mkdir(parents=True, exist_ok=True)
    log = sc.STUDY_DIR / "logs" / f"download_{a.model_key}.json"
    rec = dict(model=a.model_key, repo=m["repo"], revision=m["revision"], dest=str(dest),
               ignore=m["ignore"], started_at_utc=utc(), status="running")
    t0 = time.monotonic()
    try:
        info = HfApi().model_info(m["repo"], revision=m["revision"], files_metadata=True)
        assert info.sha == m["revision"], (info.sha, m["revision"])
        want = {f.rfilename: f.size for f in info.siblings
                if not any(fnmatch(f.rfilename, p) for p in m["ignore"])}
        snapshot_download(m["repo"], revision=m["revision"], local_dir=str(dest),
                          ignore_patterns=m["ignore"] or None, max_workers=a.workers)
        bad = []
        total = 0
        for name, size in want.items():
            p = dest / name
            if not p.exists():
                bad.append([name, "missing"])
            elif size is not None and p.stat().st_size != size:
                bad.append([name, p.stat().st_size, size])
            else:
                total += p.stat().st_size
        rec.update(files_expected=len(want), files_bad=bad, bytes=total,
                   status="ok" if not bad else "size_mismatch")
    except Exception as e:  # keep the failure as evidence
        rec.update(status="failed", error=f"{type(e).__name__}: {e}")
    rec.update(finished_at_utc=utc(), elapsed_seconds=round(time.monotonic() - t0, 1))
    log.write_text(json.dumps(rec, indent=1))
    print(json.dumps({k: v for k, v in rec.items() if k != "files_bad"}), flush=True)
    return 0 if rec["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
