#!/usr/bin/env python3
"""dp_ slot: compare two answer files (out_*.json of cc_collect.py / dp_cc.py). stats() is copied unchanged from
~/w/cc/cc_compare.py (2026-11-26): a fork is the first position where the token lists differ (or the shorter length);
logprob differences are taken before the fork.

    python dp_compare.py <out json> <label> <a.json> <b.json> [<label> <a.json> <b.json> ...]
"""
import json
import sys
from pathlib import Path


def stats(xs, ys):
    same, same_text, forks, per, maxes, n, bit, errors = 0, 0, [], [], [], 0, 0, 0
    for i, (a, b) in enumerate(zip(xs, ys)):
        if not (a["ok"] and b["ok"]):
            errors += 1
            per.append({"i": i, "error": True})
            continue
        m = min(len(a["tokens"]), len(b["tokens"]))
        first = next((k for k in range(m) if a["tokens"][k] != b["tokens"][k]), None)
        if first is None and len(a["tokens"]) == len(b["tokens"]):
            same += 1
            fork = None
        else:
            fork = m if first is None else first
            forks.append([i, fork])
        same_text += a["text"] == b["text"]
        upto = m if first is None else first
        d = [abs(float(x) - float(y)) for x, y in zip(a["logprobs"][:upto], b["logprobs"][:upto])
             if x is not None and y is not None]
        mx = max(d) if d else None
        if d:
            maxes.append(mx)
            n += len(d)
            bit += sum(1 for x in d if x == 0.0)
        per.append({"i": i, "fork": fork, "len_a": len(a["tokens"]), "len_b": len(b["tokens"]), "max_d": mx,
                    "bit_equal": sum(1 for x in d if x == 0.0), "compared": len(d),
                    "lp_at_fork_a": a["logprobs"][fork] if fork is not None and fork < len(a["logprobs"]) else None,
                    "lp_at_fork_b": b["logprobs"][fork] if fork is not None and fork < len(b["logprobs"]) else None})
    return {"requests": len(xs), "errors": errors, "same_tokens": same, "same_text": same_text,
            "fork_positions_i_pos": forks, "tokens_compared_before_fork": n, "bit_equal_logprobs": bit,
            "max_abs_dlogprob_before_fork": max(maxes) if maxes else None,
            "per_prompt": per}


def main():
    out, rest = Path(sys.argv[1]), sys.argv[2:]
    lines = []
    for k in range(0, len(rest), 3):
        label, fa, fb = rest[k:k + 3]
        row = {"label": label, "a": fa, "b": fb}
        pa, pb = Path(fa), Path(fb)
        if not (pa.is_file() and pb.is_file()):
            row["missing"] = [str(p) for p in (pa, pb) if not p.is_file()]
        else:
            da, db = json.loads(pa.read_text(encoding="utf-8")), json.loads(pb.read_text(encoding="utf-8"))
            row["params_a"], row["params_b"] = da.get("params"), db.get("params")
            row.update(stats(da["answers"], db["answers"]))
        lines.append(row)
        print(json.dumps({x: row.get(x) for x in ("label", "missing", "requests", "errors", "same_tokens", "same_text",
                                                  "fork_positions_i_pos", "tokens_compared_before_fork",
                                                  "bit_equal_logprobs", "max_abs_dlogprob_before_fork") if x in row}))
    out.write_text("".join(json.dumps(r) + "\n" for r in lines), encoding="utf-8")


if __name__ == "__main__":
    main()
