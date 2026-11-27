"""Dump the small metadata json members of one NEFF (read-only). Usage: sudo python3 neffdump.py <neff>"""
import io, sys, tarfile

data = open(sys.argv[1], "rb").read()
print("header:", data[:1024].replace(b"\x00", b"").decode("latin-1")[:600])
off = data.find(b"\x1f\x8b\x08")
tf = tarfile.open(fileobj=io.BytesIO(data[off:]), mode="r:gz")
for name in ("neff.json", "info.json", "kelf-0.json"):
    b = tf.extractfile(name).read()
    print(f"### {name} ({len(b)} bytes)")
    print(b[:2500].decode("utf-8", "replace"))
