"""Compare two kv_tokdump.py outputs: per prompt, identical id sequence or index of first differing token.
  kv_tokcmp.py <control json> <variant json>"""
import json
import sys

a = json.load(open(sys.argv[1]))
b = json.load(open(sys.argv[2]))
same = 0
tot_a = tot_b = 0
lines = []
for ra, rb in zip(a["rows"], b["rows"]):
    ia, ib = ra.get("ids", []), rb.get("ids", [])
    tot_a += len(ia)
    tot_b += len(ib)
    if ia == ib and ia:
        same += 1
        lines.append(f"prompt {ra['i']}: identical ({len(ia)} tokens)")
    else:
        k = next((j for j, (x, y) in enumerate(zip(ia, ib)) if x != y), min(len(ia), len(ib)))
        lines.append(f"prompt {ra['i']}: differ at token {k} (len {len(ia)} vs {len(ib)})")
print("\n".join(lines))
print(f"TOKCMP {sys.argv[1]} vs {sys.argv[2]}: identical {same}/{len(a['rows'])} prompts; "
      f"tokens {tot_a} vs {tot_b}; errors {a['errors']} / {b['errors']}")
