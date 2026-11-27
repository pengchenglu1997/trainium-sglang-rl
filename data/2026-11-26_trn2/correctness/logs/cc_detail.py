#!/usr/bin/env python3
"""Per-prompt detail of a pair: position of the first non-bit-equal logprob, fork position, and at the fork the
logprob each server gave its own token (small gap = near-tie)."""
import json
import sys
from pathlib import Path

RES = Path.home() / "w" / "cc" / "results"
a, b, s = sys.argv[1], sys.argv[2], sys.argv[3]
A = json.loads((RES / f"out_{a}_{s}.json").read_text())["answers"]
B = json.loads((RES / f"out_{b}_{s}.json").read_text())["answers"]
for i, (x, y) in enumerate(zip(A, B)):
    m = min(len(x["tokens"]), len(y["tokens"]))
    fork = next((k for k in range(m) if x["tokens"][k] != y["tokens"][k]), None)
    upto = m if fork is None else fork
    first_ne = next((k for k in range(upto) if x["logprobs"][k] != y["logprobs"][k]), None)
    row = {"i": i, "fork": fork, "first_unequal_logprob_pos": first_ne,
           "lp0": [x["logprobs"][0], y["logprobs"][0]] if m else None}
    if fork is not None:
        row["at_fork"] = {"tok_a": x["tokens"][fork], "lp_a": x["logprobs"][fork],
                          "tok_b": y["tokens"][fork], "lp_b": y["logprobs"][fork]}
    print(json.dumps(row))
