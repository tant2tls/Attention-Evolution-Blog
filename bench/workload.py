"""Shared public workload: raw token-ID prompts cut from public-domain Project Gutenberg texts.

The corpus (War and Peace, Moby Dick, Pride and Prejudice) is downloaded once to
WORK_ROOT/workload and verified by SHA-256. Each model tokenizes the same text with its own
tokenizer (no special tokens, no chat template). Prompt i of repetition r at length L is a cyclic
slice starting at a deterministic offset, so requests are distinct and repetitions rebuild fresh
requests. Actual token counts and a hash of every prompt are stored with each raw run.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

import study_config as sc

CORPUS = [
    ("https://www.gutenberg.org/cache/epub/2600/pg2600.txt", "pg2600.txt",
     "2d5bb2ad5f422765e714617e21fa31bbaf8958aa79682c86fca6660fcc5d1b2b"),
    ("https://www.gutenberg.org/cache/epub/2701/pg2701.txt", "pg2701.txt",
     "b5e88b71a1cda068e9ab1d94344c9d932338ba6da1a66def97a1aecd9cfcc87d"),
    ("https://www.gutenberg.org/cache/epub/1342/pg1342.txt", "pg1342.txt",
     "3f6bb9d6f78e0293b56acd4714dd68cb7d6d1d293402031ce9d5a216bcaf9d75"),
]
STRIDE = 104729  # prime offset step between distinct prompts
WORKLOAD_VERSION = "gutenberg3-cyclic-v1"


def corpus_text() -> str:
    d = sc.WORK_ROOT / "workload"
    parts = []
    for _url, name, sha in CORPUS:
        b = (d / name).read_bytes()
        got = hashlib.sha256(b).hexdigest()
        if got != sha:
            raise RuntimeError(f"corpus hash mismatch {name}: {got}")
        parts.append(b.decode("utf-8"))
    return "\n\n".join(parts)


def token_stream(model_key: str, tokenizer) -> np.ndarray:
    cache = sc.WORK_ROOT / "workload" / f"tokens_{model_key}.npy"
    if cache.exists():
        return np.load(cache)
    ids = tokenizer.encode(corpus_text(), add_special_tokens=False)
    arr = np.asarray(ids, dtype=np.int64)
    np.save(cache, arr)
    return arr


def prompt(stream: np.ndarray, length: int, index: int, rep: int, salt: int = 0) -> list[int]:
    n = len(stream)
    start = ((rep * 64 + index + salt * 997) * STRIDE) % n
    idx = (np.arange(length, dtype=np.int64) + start) % n
    return stream[idx].tolist()


def prompt_hash(ids: list[int]) -> str:
    return hashlib.sha256(np.asarray(ids, dtype=np.int64).tobytes()).hexdigest()[:16]


def describe() -> dict:
    return dict(version=WORKLOAD_VERSION, stride=STRIDE,
                corpus=[dict(url=u, file=f, sha256=s) for u, f, s in CORPUS],
                rule="prompt(L,i,rep)=stream[(start+k) % N], start=((rep*64+i+salt*997)*STRIDE) % N; "
                     "no special tokens, no chat template")


if __name__ == "__main__":
    print(json.dumps(describe(), indent=1))
