#!/usr/bin/env python3
"""Does the multi-core LoRA trainer compute the same update as one core, and how much faster?

    python 56_ptrainer_probe.py --model DIR --train-jsonl FILE --results DIR --tag pt4 \
        --core 32 --worker-cores 33,34,35 [--sequences 128] [--compile]

The same seeded rows (GSM8K questions with their reference solutions as "answers", advantages
+1 / -1 in turn) go through two updates. Recorded per update: seconds, micro-batches, processes,
the loss, the mean token log-probability, and after the first update a checksum of the LoRA values
(sum of absolute values and of squares), which a run with other worker counts must reproduce up to
floating-point order."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rl_common as C  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--train-jsonl", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--core", required=True)
    ap.add_argument("--worker-cores", default="")
    ap.add_argument("--sequences", type=int, default=128)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--len-buckets", default=None, help="total lengths, ascending, last = 576 (e.g. 320,448,576)")
    ap.add_argument("--answer-window", action="store_true", help="vocabulary logits only at answer positions")
    ap.add_argument("--updates", type=int, default=3)
    a = ap.parse_args()
    res = C.Results(a.results, f"56_ptrainer_{a.tag}")
    tr = None
    try:
        import trainer as T
        T.set_neuron_env(a.core, compile_train=a.compile)
        import ptrainer
        cores = [c for c in a.worker_cores.split(",") if c]
        t0 = time.time()
        tr = ptrainer.ParallelTrainer(a.model, worker_cores=cores, lr=1e-4, max_prompt=320, max_completion=256,
                                      micro_bs=4, seed=1, compile_train=a.compile,
                                      len_buckets=[int(x) for x in a.len_buckets.split(",")] if a.len_buckets else None,
                                      answer_window=a.answer_window)
        res.write(event="ready", processes=len(cores) + 1, seconds=round(time.time() - t0, 1), compile=a.compile)
        items = []
        with open(a.train_jsonl, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if len(items) >= a.sequences:
                    break
                r = json.loads(line)
                if tr.prompt_tokens(r["question"]) <= 320:
                    items.append((r["question"], r["answer"], 1.0 if i % 2 == 0 else -1.0))
        for u in range(a.updates):
            out = tr.update(items, temperature=0.7)
            torch = tr.torch
            vals = [p.detach().float().cpu() for p in tr.params]
            res.write(event="update", n=u, **{k: v for k, v in out.items()},
                      lora_abs_sum=round(float(sum(v.abs().sum() for v in vals)), 4),
                      lora_sq_sum=round(float(sum((v * v).sum() for v in vals)), 6), peak_bytes=tr.peak_bytes())
    except BaseException as e:  # noqa: BLE001
        res.fail("ptrainer", e)
        raise
    finally:
        try:
            if tr is not None:
                tr.close()
        except Exception:  # noqa: BLE001
            pass
        res.done()


if __name__ == "__main__":
    main()
