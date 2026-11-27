"""Print compiler arguments embedded in NEFF files (read-only). Usage: sudo python3 neffargs.py <cache dir> [max files]"""
import io, os, re, sys, tarfile, json

root = sys.argv[1]
limit = int(sys.argv[2]) if len(sys.argv) > 2 else 100
files = []
for d, _, fs in os.walk(os.path.join(root, "neff")):
    for f in fs:
        if f.endswith(".neff"):
            p = os.path.join(d, f)
            files.append((os.path.getsize(p), p))
files.sort(reverse=True)
pat = re.compile(rb"(neuronx-cc[^\x00]{0,3000}|--model-type[^\x00]{0,2000})")
for size, p in files[:limit]:
    data = open(p, "rb").read()
    print(f"### {p} size={size}")
    off = data.find(b"\x1f\x8b\x08")
    members = []
    if off >= 0:
        try:
            tf = tarfile.open(fileobj=io.BytesIO(data[off:]), mode="r:gz")
            for m in tf.getmembers():
                members.append(m.name)
                if m.isfile() and (m.name.endswith(".json") or "kelp" in m.name or m.name.endswith(".txt")):
                    b = tf.extractfile(m).read()
                    for mm in pat.findall(b):
                        print(f"  [{m.name}] {mm[:1500].decode('utf-8','replace')}")
                    if b.find(b"compiler") >= 0 and m.name.endswith(".json"):
                        try:
                            j = json.loads(b)
                            def walk(o, path=""):
                                if isinstance(o, dict):
                                    for k, v in o.items():
                                        if "compil" in k.lower() or "arg" in k.lower() or "flag" in k.lower() or "version" in k.lower():
                                            print(f"  [{m.name}] {path}/{k} = {str(v)[:1500]}")
                                        walk(v, f"{path}/{k}")
                                elif isinstance(o, list):
                                    for i, v in enumerate(o[:50]):
                                        walk(v, f"{path}[{i}]")
                            walk(j)
                        except Exception as e:
                            print(f"  [{m.name}] json error {e}")
        except Exception as e:
            print(f"  tar error at {off}: {e}")
    print(f"  members: {members[:40]}")
    for mm in pat.findall(data[:off] if off > 0 else data[:4096]):
        print(f"  [header] {mm[:1500].decode('utf-8','replace')}")
