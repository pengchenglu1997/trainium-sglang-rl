#!/usr/bin/env python3
"""i37: same-server repeat check of out_<name>_t0.json: for each of the repeated first prompts, compare the repeat with
the first answer (tokens, sampled logprobs per position, position-0 top-k vector).

    python i37_repeat.py <out.json> [<out.json> ...]
"""
import json
import sys

for path in sys.argv[1:]:
    o = json.load(open(path, encoding="utf-8"))
    for k, r in enumerate(o.get("repeat_first3") or []):
        a = o["answers"][k]
        n = min(len(a["tokens"]), len(r["tokens"]))
        fork = next((j for j in range(n) if a["tokens"][j] != r["tokens"][j]), None)
        upto = fork if fork is not None else n
        diffs = [r["logprobs"][j] - a["logprobs"][j] for j in range(upto)]
        first_neq = next((j for j, v in enumerate(diffs) if v != 0.0), None)
        da = dict((t, l) for t, l in a["top"][0])
        d0 = [l - da[t] for t, l in r["top"][0] if t in da]
        print(json.dumps({"file": path.split("/")[-1], "prompt": k, "same_tokens": a["tokens"] == r["tokens"],
                          "fork": fork, "first_unequal_logprob_pos": first_neq,
                          "unequal_positions": sum(1 for v in diffs if v != 0.0), "compared": upto,
                          "max_abs_d": max((abs(v) for v in diffs), default=0.0),
                          "pos0_top_bit_equal": sum(1 for v in d0 if v == 0.0), "pos0_top_n": len(d0),
                          "pos0_top_max_abs_d": max((abs(v) for v in d0), default=0.0)}))
