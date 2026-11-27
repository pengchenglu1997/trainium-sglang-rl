"""List the NKI config-cache entries in a vllm cache folder: kernel name and the .colz path each one points at.
usage: sudo python3 nki_entries.py <cache dir>"""
import base64, glob, json, os, sys

d = os.path.join(sys.argv[1], "neuron", "compile_cache", "nki")
files = sorted(glob.glob(os.path.join(d, "*.json")))
print(f"dir {d}: {len(files)} json entries")
for f in files:
    try:
        data = json.load(open(f))
        cfg = json.loads(base64.b64decode(data["dumped_config"]))
        print(os.path.basename(f), cfg.get("func_name"), cfg.get("klir_binary", {}).get("binary"))
    except Exception as e:  # report and go on
        print(os.path.basename(f), "unreadable:", type(e).__name__, e)
