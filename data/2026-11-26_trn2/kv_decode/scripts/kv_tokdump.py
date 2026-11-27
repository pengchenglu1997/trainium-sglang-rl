"""20-prompt temperature-0 token dump for a token comparison between two server starts (kv slot, 2026-11-26).
  kv_tokdump.py <base url> <served model> <gsm8k train.jsonl> <out json>
Prompts: the first 20 questions of the jsonl, sent one at a time to SGLang's native /generate with temperature 0,
max_new_tokens 128, return_logprob true (the output token ids are read from meta_info.output_token_logprobs).
Last line: LLMBENCH_DONE tokdump ... (so 76_serve_and_bench.py counts it as done) or TOKDUMP_FAILED."""
import json
import sys
import urllib.request

url, served, jsonl, out = sys.argv[1:5]
rows = []
with open(jsonl) as f:
    for line in f:
        if line.strip():
            rows.append(json.loads(line)["question"])
        if len(rows) == 20:
            break
res = []
errors = 0
for i, q in enumerate(rows):
    body = {"text": "Question: " + q + "\nAnswer:", "sampling_params": {"temperature": 0.0, "max_new_tokens": 128},
            "return_logprob": True}
    req = urllib.request.Request(url.rstrip("/") + "/generate", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            d = json.loads(r.read())
        mi = d.get("meta_info", {})
        ids = [t[1] for t in mi.get("output_token_logprobs", [])]
        res.append({"i": i, "text": d.get("text"), "ids": ids, "completion_tokens": mi.get("completion_tokens")})
    except Exception as e:  # noqa: BLE001
        errors += 1
        res.append({"i": i, "error": f"{type(e).__name__}: {e}"[:300]})
with open(out, "w") as f:
    json.dump({"url": url, "served": served, "n": len(res), "errors": errors, "rows": res}, f)
if errors:
    print(f"TOKDUMP_FAILED errors={errors} out={out}")
    sys.exit(1)
print(f"LLMBENCH_DONE tokdump out={out} n={len(res)} tokens={sum(len(r['ids']) for r in res)}")
