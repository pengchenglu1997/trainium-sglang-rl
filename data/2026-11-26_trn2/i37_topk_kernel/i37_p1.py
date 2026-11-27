#!/usr/bin/env python3
"""i37: prompt 1 (index 1) across servers: first unequal logprob position of every answer and repeat against
r1_koff's first answer.   python i37_p1.py <results dir> <name> [...]"""
import json
import sys

d = sys.argv[1]
ref = json.load(open(f"{d}/out_r1_koff_t0.json"))["answers"][1]["logprobs"]
for n in sys.argv[2:]:
    o = json.load(open(f"{d}/out_{n}_t0.json"))
    for kind, a in (("first", o["answers"][1]), ("repeat", o["repeat_first3"][1])):
        lp = a["logprobs"]
        neq = [j for j in range(min(len(lp), len(ref))) if lp[j] != ref[j]]
        print(json.dumps({"server": n, "answer": kind, "unequal_vs_r1_koff_first": len(neq),
                          "first_unequal": neq[0] if neq else None}))
