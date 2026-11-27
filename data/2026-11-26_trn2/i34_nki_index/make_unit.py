"""Build two test cache folders for the keep/remove branches of nki_index_check.py, from one real index entry.
usage: python3 make_unit.py <real entry json> <out root>
  <out root>/keep/neuron/compile_cache/nki/<entry>: klir_binary.binary rewritten to /cache/colz/k.colz, file created
  <out root>/remove/...: same entry unchanged (its /var/tmp path) plus the rewritten one"""
import base64, json, os, sys

src, root = sys.argv[1], sys.argv[2]
data = json.load(open(src))
cfg = json.loads(base64.b64decode(data["dumped_config"]))
for case in ("keep", "remove"):
    d = os.path.join(root, case, "neuron", "compile_cache", "nki")
    os.makedirs(d, exist_ok=True)
    os.makedirs(os.path.join(root, case, "colz"), exist_ok=True)
    open(os.path.join(root, case, "colz", "k.colz"), "wb").write(b"test")
    c2 = dict(cfg); c2["klir_binary"] = dict(cfg["klir_binary"]); c2["klir_binary"]["binary"] = "/cache/colz/k.colz"
    d2 = dict(data); d2["dumped_config"] = base64.b64encode(json.dumps(c2).encode()).decode()
    json.dump(d2, open(os.path.join(d, "rewritten.json"), "w"))
    if case == "remove":
        json.dump(data, open(os.path.join(d, "original.json"), "w"))
print("built", root)
