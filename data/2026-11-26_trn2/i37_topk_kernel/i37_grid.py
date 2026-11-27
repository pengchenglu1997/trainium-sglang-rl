#!/usr/bin/env python3
"""i37: are the position-0 logits on a bf16 grid? At temperature 0 the returned top logprobs are logit - lse, so the gap
lp(j) - lp(top1) is the logit difference. If the logits are bf16 values, each gap is a multiple of the bf16 spacing
of the logits' range (0.125 for logits in [16, 32), 0.0625 in [8, 16)), up to fp32 rounding of the subtraction.

For each file: per prompt, the fraction of the 19 gaps within 2e-4 of a multiple of 1/16 and of 1/8, and the
fraction within 2e-4 of a multiple of 1/64 (a control: a finer grid catches more by chance).

    python i37_grid.py <out.json> [...]
"""
import json
import sys


def near(x, q, tol=2e-4):
    r = (x / q) - round(x / q)
    return abs(r * q) <= tol


for path in sys.argv[1:]:
    o = json.load(open(path, encoding="utf-8"))
    f16 = f8 = f64 = n = 0
    per = []
    for a in o["answers"]:
        top = a["top"][0]
        g = [l - top[0][1] for _, l in top[1:]]
        n += len(g)
        c16 = sum(near(x, 1 / 16) for x in g)
        f16 += c16
        f8 += sum(near(x, 1 / 8) for x in g)
        f64 += sum(near(x, 1 / 64) for x in g)
        per.append(c16)
    print(json.dumps({"file": path.split("/")[-1], "gaps": n, "on_1_16_grid": f16, "on_1_8_grid": f8,
                      "on_1_64_grid": f64, "per_prompt_on_1_16": per}))
