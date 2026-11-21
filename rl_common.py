"""Shared helpers of the SGLang / Trainium RL kit (standard library only at import time).

Results files are JSON lines; a finished run ends with {"done": true}; a failed step writes a
line with "status": "fail" before it. Counts and timings only.
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

SYSTEM = "You are a careful math tutor. Solve the problem step by step."
INSTRUCTION = ("\n\nSolve it step by step. On the last line write the final answer as "
               "'#### <number>' and nothing else.")


class Results:
    def __init__(self, results_dir, name):
        d = Path(results_dir)
        d.mkdir(parents=True, exist_ok=True)
        self.path = d / f"{name}.jsonl"
        if self.path.exists():
            self.path.rename(self.path.with_name(f"{self.path.name}.prev.{time.time_ns()}"))
        self.write(event="open", pid=os.getpid(), unix=round(time.time(), 1))

    def write(self, **kv):
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(kv, ensure_ascii=False) + "\n")

    def fail(self, stage, exc):
        self.write(status="fail", stage=stage, error_type=type(exc).__name__,
                   message=str(exc)[:400])

    def done(self):
        with open(self.path, "a", encoding="utf-8") as f:
            f.write('{"done": true}\n')


def read_jsonl(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def finished_ok(path) -> bool:
    try:
        rows = read_jsonl(path)
    except OSError:
        return False
    return bool(rows) and rows[-1] == {"done": True} and not any(r.get("status") == "fail" for r in rows)


# ---------------------------------------------------------------------------
# GSM8K
# ---------------------------------------------------------------------------
_NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def to_number(text):
    """First number in the text as a float (commas, a leading $ and a trailing % or . dropped)."""
    if text is None:
        return None
    m = _NUM.search(str(text).replace("$", ""))
    if not m:
        return None
    try:
        return float(m.group().replace(",", ""))
    except ValueError:
        return None


def gold_answer(answer_field: str):
    """GSM8K gold: the number after '####' in the dataset's answer field."""
    return to_number(answer_field.split("####")[-1])


def extract_answer(completion: str):
    """Model answer: the number after the last '####'; else the last number in the text."""
    if completion is None:
        return None
    text = re.sub(r"<think>.*?</think>", "", completion, flags=re.DOTALL)
    if "####" in text:
        v = to_number(text.split("####")[-1])
        if v is not None:
            return v
    nums = _NUM.findall(text.replace("$", ""))
    if not nums:
        return None
    try:
        return float(nums[-1].replace(",", ""))
    except ValueError:
        return None


def is_correct(completion: str, gold) -> bool:
    v = extract_answer(completion)
    return v is not None and gold is not None and abs(v - gold) < 1e-6


def reward(completion: str, gold) -> float:
    """1.0 for the right final number, else 0.0 (the binary reward of the GSM8K demos)."""
    return 1.0 if is_correct(completion, gold) else 0.0


def load_gsm8k(path, limit=None):
    """Rows of a GSM8K jsonl file (fields question, answer): list of (index, question, gold)."""
    out = []
    for i, r in enumerate(read_jsonl(path)):
        g = gold_answer(r["answer"])
        if g is not None:
            out.append((i, r["question"], g))
        if limit and len(out) >= limit:
            break
    return out


def messages_for(question: str):
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": question + INSTRUCTION}]


def describe(xs):
    xs = sorted(float(x) for x in xs)
    if not xs:
        return {"n": 0}
    n = len(xs)
    med = xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2
    return {"n": n, "mean": round(sum(xs) / n, 4), "min": round(xs[0], 4),
            "median": round(med, 4), "max": round(xs[-1], 4)}


def group_advantages(rewards, eps=1e-4):
    """GRPO advantages of one group: (r - mean) / (std + eps); all zero when rewards are equal."""
    n = len(rewards)
    mu = sum(rewards) / n
    var = sum((r - mu) ** 2 for r in rewards) / n
    sd = var ** 0.5
    if sd < 1e-8:
        return [0.0] * n
    return [(r - mu) / (sd + eps) for r in rewards]
