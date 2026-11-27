#!/usr/bin/env python3
"""i37: compare two out_<name>_t0.json files (top_logprobs on).

    python i37_compare.py <a.json> <b.json> [label]

Per prompt, first generated token (position 0) and over the common prefix:
  - same tokens, bit-equal sampled logprobs
  - top-k vectors at position 0: for tokens present in both top lists, d_j = lp_b(j) - lp_a(j).
    A pure log-normalizer change gives d_j equal for all j (std 0); a logit change gives scattered d_j.
    gap_j = (lp(j) - lp(top1)) is lse-free: its difference b - a is the logit-difference part.
  - top-1 id order agreement, and same-server repeat (repeat_first3) bit-equality.
Prints one JSON line per prompt and a summary JSON line.
"""
import json
import statistics as st
import sys


def load(p):
    return json.load(open(p, encoding="utf-8"))


def main():
    a, b = load(sys.argv[1]), load(sys.argv[2])
    label = sys.argv[3] if len(sys.argv) > 3 else f"{a['name']} vs {b['name']}"
    rows = []
    for i, (x, y) in enumerate(zip(a["answers"], b["answers"])):
        if not (x["ok"] and y["ok"]):
            rows.append({"i": i, "error": True})
            continue
        ta, tb = x["top"][0], y["top"][0]
        da, db = dict((t, l) for t, l in ta), dict((t, l) for t, l in tb)
        common = [t for t, _ in ta if t in db]
        d = [db[t] - da[t] for t in common]
        top1a, top1b = ta[0][1], tb[0][1]
        gapd = [(db[t] - top1b) - (da[t] - top1a) for t in common]
        n = 0
        eq = 0
        for k in range(min(len(x["tokens"]), len(y["tokens"]))):
            if x["tokens"][k] != y["tokens"][k]:
                break
            n += 1
            eq += x["logprobs"][k] == y["logprobs"][k]
        rows.append({
            "i": i, "tok0_a": x["tokens"][0], "tok0_b": y["tokens"][0], "same_tokens_all": x["tokens"] == y["tokens"],
            "lp0_a": x["logprobs"][0], "lp0_b": y["logprobs"][0], "d_lp0": y["logprobs"][0] - x["logprobs"][0],
            "top_n_a": len(ta), "top_n_b": len(tb), "top_common": len(common),
            "same_top_order": [t for t, _ in ta] == [t for t, _ in tb],
            "top_bit_equal": sum(1 for v in d if v == 0.0),
            "d_top_mean": st.mean(d) if d else None, "d_top_std": st.pstdev(d) if len(d) > 1 else None,
            "d_top_min": min(d) if d else None, "d_top_max": max(d) if d else None,
            "d_top_pos": sum(1 for v in d if v > 0), "d_top_neg": sum(1 for v in d if v < 0),
            "gap_d_abs_max": max(abs(v) for v in gapd) if gapd else None,
            "gap_d_abs_mean": st.mean(abs(v) for v in gapd) if gapd else None,
            "prefix_compared": n, "prefix_bit_equal": eq,
        })
    for r in rows:
        print(json.dumps(r))
    ok = [r for r in rows if not r.get("error")]

    def rep(o):
        rr = o.get("repeat_first3") or []
        return sum(1 for k, z in enumerate(rr) if z.get("ok") and z["logprobs"] == o["answers"][k]["logprobs"]
                   and z["top"] == o["answers"][k]["top"]), len(rr)

    summ = {
        "summary": label, "prompts": len(rows), "errors": len(rows) - len(ok),
        "lp0_bit_equal": sum(1 for r in ok if r["d_lp0"] == 0.0),
        "same_tok0": sum(1 for r in ok if r["tok0_a"] == r["tok0_b"]),
        "same_tokens_all": sum(1 for r in ok if r["same_tokens_all"]),
        "same_top_order": sum(1 for r in ok if r["same_top_order"]),
        "mean_abs_d_lp0": st.mean(abs(r["d_lp0"]) for r in ok),
        "max_abs_d_lp0": max(abs(r["d_lp0"]) for r in ok),
        "median_d_top_std": st.median(r["d_top_std"] for r in ok if r["d_top_std"] is not None),
        "median_abs_d_top_mean": st.median(abs(r["d_top_mean"]) for r in ok if r["d_top_mean"] is not None),
        "median_gap_d_abs_mean": st.median(r["gap_d_abs_mean"] for r in ok if r["gap_d_abs_mean"] is not None),
        "prompts_top_all_bit_equal": sum(1 for r in ok if r["top_bit_equal"] == r["top_common"]),
        "prefix_bit_equal": sum(r["prefix_bit_equal"] for r in ok), "prefix_compared": sum(r["prefix_compared"] for r in ok),
        "repeat_a_bit_equal": rep(a), "repeat_b_bit_equal": rep(b),
    }
    print(json.dumps(summ))


if __name__ == "__main__":
    main()
