#!/usr/bin/env python3
"""A pass-through HTTP proxy in front of the SGLang-Neuron server that answers POST /pull_weights itself, so that
Miles's disk-delta weight sync can drive our server (which has no /pull_weights route; see 64_miles_sequence.md, gap 1).

    python 66_pull_weights_shim.py --port 30001 --upstream http://127.0.0.1:30000 \
        --base-dir /srv/models/qwen3-0.6b --host-root /srv/models --served-root /models [--listen-host 127.0.0.1]

Point Miles's rollout engine address at the shim's port instead of the server's. Miles then sends every call of one
update (pull_weights, pause_generation, flush_cache, update_weights_from_disk, continue_generation, model_info) to the
shim. Every route except POST /pull_weights is forwarded to --upstream unchanged (method, path, query string, headers,
body; the answer, including a chunked or streamed one, is passed back in pieces as it arrives). Two exceptions to
"unchanged", both on purpose: hop-by-hop headers are dropped as HTTP requires, and the model_path field of a
POST /update_weights_from_disk body is rewritten from a host path to the path the server sees (next paragraph).

Path mapping. Miles and this shim run on the host and see host paths (--update-weight-local-checkpoint-dir, the delta
directory); the server runs in a docker container where MODELS_ROOT is mounted at /models (srv_sglang_neuron.sh).
--host-root is the host folder and --served-root its mount point in the container. In a POST /update_weights_from_disk
body, a model_path equal to or below --host-root is rewritten to the same place below --served-root, and the body is
re-serialised as JSON; a body without such a model_path is forwarded byte for byte. The local checkpoint directory
therefore has to lie below --host-root, and --host-root has to be mounted into the container. Give both options or
neither. The paths in a /pull_weights body are used as they are (host paths): the shim itself reads and writes them.

What /pull_weights does here. The real engine side is sglang.srt.weight_sync.local_checkpoint.pull in the branch
sglang-miles of github.com/sgl-project/sglang (commit de700e693026 when read on 22 November); this file follows it and
differs only where listed below.
  * Request: JSON {"local_checkpoint_dir", "source_dir", "target_version"}. A missing or mistyped field answers HTTP 422
    (FastAPI's status for a bad body). Answer: HTTP 200 {"success": true, "message": "Success."}, or HTTP 400
    {"success": false, "message": <reason>}. Miles raises on HTTP >= 400 or "success": false.
  * Base: --base-dir (a host path), the engine's model_path. The Miles payload does not carry it (the real engine uses
    its own model_path).
  * target_version 0: local_checkpoint_dir becomes a copy of the files of --base-dir, marker version 0.
  * target_version N > 0: the versions after the one the local directory holds are applied in order, each read from
    <source_dir>/weight_vNNNNNN/. A version is a delta when its model.safetensors.index.json metadata has
    "delta_encoding", otherwise a full checkpoint, which replaces the local directory (the engine does the same).
    Delta: for each tensor of the version's model-*.safetensors the bytes are zstd-decompressed and either XORed onto
    the local tensor bytes (encoding xor) or scattered into them (encoding overwrite: u4 count, u4 positions, new
    values); the result is checksummed (xxh3-128, blake3 or adler32, as the index metadata says) and compared with the
    checksum stored in the delta file's safetensors metadata. Lineage: the index metadata "version" must be the
    directory's version, "base_version" the version the local directory holds, compression "zstd".
  * Atomic: everything is built in a temporary directory beside local_checkpoint_dir (same filesystem) and swapped in
    with renames at the end (old directory renamed aside, new one renamed into place, old one deleted). Any refusal
    deletes the temporary directory and leaves local_checkpoint_dir as it was. The two renames are not one atomic
    operation: for a moment local_checkpoint_dir does not exist. Nothing reads it then, because the server loads it
    only in /update_weights_from_disk, which Miles sends after the pull has answered.
  * Marker: <local_checkpoint_dir>/.weight_sync/state.json {"version": "000003", "base_dir": ...} (the engine writes
    the version only). A pull to the version the marker holds is a no-op that touches nothing. Concurrent pulls of one
    directory are serialised by an in-process lock.
  * Refused (HTTP 400, directory untouched): a version directory missing from the chain (a gap), a delta whose
    version or base_version does not match, a checksum mismatch, a corrupt or wrongly sized zstd stream, a delta
    tensor the local checkpoint does not have, a delta file set that differs from the index's weight_map, an unknown
    encoding, compression or checksum name, target_version below the marker's version (above 0).
Differences from the engine, all stricter or deliberate: (1) target_version 0 on a directory whose marker is above 0
resets it to the base (the engine keeps it, so a restarted Miles run would hit a base_version mismatch at its first
delta); (2) target_version below the marker (above 0) is refused (the engine does nothing); (3) a delta's declared
version and file set are checked; (4) a local directory that exists, is not empty and has no marker is refused unless
--adopt-unmarked is given, and local_checkpoint_dir may not equal or contain or lie inside --base-dir or source_dir
(the engine would overwrite it); (5) a marker written by another --base-dir is refused; (6) the apply runs on a copy and
swaps (the engine patches in place under an flock), so a pull copies the whole checkpoint: seconds for 0.6B, not
measured here; (7) only files, not sub-directories, are copied from the base, as the engine does.
Assumption not checked against the engine: the files' mode. The temporary directory is made 0755 and copied files keep
their source mode, so a container user that is not the owner can read the result.

The shim has no authentication, as the engine's route has none by default: keep --listen-host on 127.0.0.1 unless the
trainer is on another machine."""
from __future__ import annotations

import argparse
import glob
import http.client
import json
import os
import shutil
import struct
import sys
import threading
import time
import urllib.parse
import zlib
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

SYNC_DIR = ".weight_sync"
HOP_BY_HOP = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "proxy-connection", "te",
              "trailer", "trailers", "transfer-encoding", "upgrade", "expect"}
CHUNK = 64 * 1024


class PullError(Exception):
    """A refusal: the directory was not changed. The message goes back to Miles."""


# ---------------------------------------------------------------------------
# checksums and zstd (the algorithms Miles's miles/utils/disk_delta.py names)
# ---------------------------------------------------------------------------
class _Adler32:
    def __init__(self):
        self._value = 1

    def update(self, data) -> None:
        self._value = zlib.adler32(data, self._value)

    def hexdigest(self) -> str:
        return f"{self._value:08x}"


def _new_hasher(algorithm: str):
    if algorithm == "xxh3-128":
        import xxhash
        return xxhash.xxh3_128()
    if algorithm == "blake3":
        import blake3
        return blake3.blake3()
    if algorithm == "adler32":
        return _Adler32()
    raise PullError(f"unknown checksum algorithm {algorithm!r}")


def checksum(algorithm: str, buf) -> str:
    h = _new_hasher(algorithm)
    h.update(buf)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# the checkpoint directory
# ---------------------------------------------------------------------------
def version_dir(source_dir: str, version: int) -> str:
    return os.path.join(source_dir, f"weight_v{version:06d}")


def read_marker(local_dir: str):
    """(version, base_dir) of the marker, or (None, None) when there is none."""
    try:
        with open(os.path.join(local_dir, SYNC_DIR, "state.json"), encoding="utf-8") as f:
            d = json.load(f)
        return int(d["version"]), d.get("base_dir")
    except FileNotFoundError:
        return None, None
    except (ValueError, KeyError, TypeError) as e:
        raise PullError(f"unreadable marker in {local_dir}: {e}")


def write_marker(ckpt_dir: str, version: int, base_dir: str) -> None:
    sync = os.path.join(ckpt_dir, SYNC_DIR)
    os.makedirs(sync, exist_ok=True)
    path = os.path.join(sync, "state.json")
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        json.dump({"version": f"{version:06d}", "base_dir": base_dir}, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(path + ".tmp", path)


def _is_delta(vdir: str) -> bool:
    if not os.path.isdir(vdir):
        raise PullError(f"published weight version missing: {vdir}")
    try:
        with open(os.path.join(vdir, "model.safetensors.index.json"), encoding="utf-8") as f:
            return "delta_encoding" in json.load(f).get("metadata", {})
    except FileNotFoundError:
        if not glob.glob(os.path.join(vdir, "*.safetensors")):
            raise PullError(f"{vdir} has neither an index nor safetensors files (not published yet?)")
        return False   # an ordinary checkpoint, as in the engine
    except ValueError as e:
        raise PullError(f"unreadable index in {vdir}: {e}")


def _copy_files(src: str, dst: str) -> None:
    n = 0
    for entry in os.scandir(src):
        if entry.is_file():
            shutil.copy2(entry.path, os.path.join(dst, entry.name))
            n += 1
    for entry in os.scandir(src):   # a truncated copy must fail loud
        if entry.is_file() and os.path.getsize(os.path.join(dst, entry.name)) != entry.stat().st_size:
            raise PullError(f"size mismatch copying {entry.name} from {src}")
    if n == 0:
        raise PullError(f"no files in {src}")


def _tensor_locations(ckpt_dir: str) -> dict:
    """tensor name -> (file, byte offset, nbytes)."""
    loc = {}
    for path in glob.glob(os.path.join(ckpt_dir, "*.safetensors")):
        with open(path, "rb") as f:
            (hl,) = struct.unpack("<Q", f.read(8))
            header = json.loads(f.read(hl))
        for name, info in header.items():
            if name == "__metadata__":
                continue
            b, e = info["data_offsets"]
            loc[name] = (path, 8 + hl + b, e - b)
    return loc


def _read_delta_items(vdir: str, weight_map: dict) -> list:
    """(name, compressed bytes, want checksum) for every tensor of every delta file of the version."""
    items, seen = [], set()
    for path in sorted(glob.glob(os.path.join(vdir, "*.safetensors"))):
        with open(path, "rb") as f:
            blob = f.read()
        if len(blob) < 8:
            raise PullError(f"{path}: too short")
        (hl,) = struct.unpack("<Q", blob[:8])
        if 8 + hl > len(blob):
            raise PullError(f"{path}: header longer than the file")
        header = json.loads(blob[8:8 + hl])
        want = header.get("__metadata__", {})
        view = memoryview(blob)
        for name, info in header.items():
            if name == "__metadata__":
                continue
            b, e = info["data_offsets"]
            if not (0 <= b <= e <= len(blob) - 8 - hl):
                raise PullError(f"{path}: tensor {name!r} points outside the file")
            if name in seen:
                raise PullError(f"tensor {name!r} appears twice in {vdir}")
            seen.add(name)
            items.append((name, view[8 + hl + b:8 + hl + e], want.get(name)))
    if seen != set(weight_map):
        diff = sorted(seen ^ set(weight_map))[:5]
        raise PullError(f"files of {vdir} do not match the index weight_map (e.g. {diff})")
    return items


def apply_delta(ckpt_dir: str, vdir: str, version: int, applied: int, workers: int) -> None:
    """Apply one published delta version to the checkpoint in ckpt_dir (a private copy), verifying every tensor."""
    import numpy as np   # third-party imports only here, so the proxy starts without them
    import zstandard
    try:
        with open(os.path.join(vdir, "model.safetensors.index.json"), encoding="utf-8") as f:
            index = json.load(f)
        meta = index["metadata"]
        v, base_v = int(meta["version"]), int(meta["base_version"])
        encoding, algorithm, compression = meta["delta_encoding"], meta["checksum_format"], meta["compression_format"]
    except (OSError, ValueError, KeyError, TypeError) as e:
        raise PullError(f"bad index in {vdir}: {type(e).__name__}: {e}")
    if v != version:
        raise PullError(f"{vdir} declares version {v}, expected {version}")
    if base_v != applied:
        raise PullError(f"out-of-order delta: local at {applied}, delta {version} builds on {base_v}")
    if compression != "zstd":
        raise PullError(f"compression {compression!r} not supported")
    if encoding not in ("xor", "overwrite"):
        raise PullError(f"delta encoding {encoding!r} not supported")
    _new_hasher(algorithm)   # unknown algorithm -> refuse before touching anything
    locations = _tensor_locations(ckpt_dir)
    items = _read_delta_items(vdir, index.get("weight_map", {}))
    for name, _, _ in items:
        if name not in locations:
            raise PullError(f"delta tensor {name!r} is not in the local checkpoint")

    def apply_one(item):
        name, compressed, want = item
        path, offset, nbytes = locations[name]
        try:
            raw = zstandard.ZstdDecompressor().decompress(compressed, max_output_size=4 + 5 * nbytes + 1)
        except zstandard.ZstdError as e:
            return name, f"zstd: {e}"
        delta = np.frombuffer(raw, dtype=np.uint8)
        with open(path, "r+b") as f:
            f.seek(offset)
            region = np.frombuffer(bytearray(f.read(nbytes)), dtype=np.uint8)
            if region.size != nbytes:
                return name, "local tensor is shorter than its header says"
            if encoding == "xor":
                if delta.size != nbytes:
                    return name, f"xor delta has {delta.size} bytes, tensor has {nbytes}"
                region ^= delta
            else:
                if delta.size < 4:
                    return name, "overwrite delta shorter than its count field"
                count = int.from_bytes(delta[:4].tobytes(), "little")
                if delta.size != 4 + 4 * count + count:
                    return name, f"overwrite delta size {delta.size} does not fit count {count}"
                pos = np.frombuffer(delta[4:4 + 4 * count].tobytes(), dtype="<u4")
                if count and int(pos.max()) >= nbytes:
                    return name, "overwrite position outside the tensor"
                region[pos] = delta[4 + 4 * count:]
            if want is None or checksum(algorithm, region) != want:
                return name, "checksum mismatch"
            f.seek(offset)
            f.write(region.tobytes())
        return name, None

    with ThreadPoolExecutor(max_workers=workers) as pool:
        bad = [(n, why) for n, why in pool.map(apply_one, items) if why]
    if bad:
        raise PullError(f"{len(bad)} tensors failed after applying {vdir}: "
                        + "; ".join(f"{n}: {why}" for n, why in sorted(bad)[:10]))


# ---------------------------------------------------------------------------
# /pull_weights
# ---------------------------------------------------------------------------
def _inside(a: str, b: str) -> bool:
    """a is b or below b (real paths)."""
    a, b = os.path.realpath(a), os.path.realpath(b)
    try:
        return os.path.commonpath([a, b]) == b
    except ValueError:
        return False


class Puller:
    def __init__(self, base_dir: str, adopt_unmarked: bool = False, workers: int | None = None, log=None):
        self.base_dir = os.path.abspath(base_dir)
        self.adopt_unmarked = adopt_unmarked
        self.workers = workers or min(8, os.cpu_count() or 4)
        self.log = log or (lambda msg: None)
        self.stats = {"pulls": 0, "noops": 0, "applied_versions": 0, "refused": 0}
        self._locks: dict[str, threading.Lock] = {}
        self._locks_guard = threading.Lock()

    def _lock(self, path: str) -> threading.Lock:
        with self._locks_guard:
            return self._locks.setdefault(os.path.realpath(path), threading.Lock())

    def pull(self, local_dir: str, source_dir: str, target: int) -> str:
        if not (os.path.isabs(local_dir) and os.path.isabs(source_dir)):
            raise PullError("local_checkpoint_dir and source_dir must be absolute paths")
        local = os.path.abspath(os.path.normpath(local_dir))
        source = os.path.abspath(os.path.normpath(source_dir))
        if target < 0:
            raise PullError("target_version must not be negative")
        if _inside(local, self.base_dir) or _inside(self.base_dir, local):
            raise PullError("local_checkpoint_dir may not be, contain or lie inside the base directory")
        if _inside(local, source) or _inside(source, local):
            raise PullError("local_checkpoint_dir may not be, contain or lie inside source_dir")
        with self._lock(local):
            try:
                msg = self._pull_locked(local, source, target)
            except PullError:
                self.stats["refused"] += 1
                raise
            except Exception as e:   # noqa: BLE001 - anything else is a refusal too, never a half-written directory
                self.stats["refused"] += 1
                raise PullError(f"{type(e).__name__}: {e}")
            self.stats["pulls"] += 1
            return msg

    def _pull_locked(self, local: str, source: str, target: int) -> str:
        applied, marker_base = read_marker(local)
        if applied is None:
            if os.path.exists(local) and (not os.path.isdir(local) or os.listdir(local)) and not self.adopt_unmarked:
                raise PullError(f"{local} exists, is not empty and has no {SYNC_DIR}/state.json marker; "
                                "remove it or start the shim with --adopt-unmarked")
        elif marker_base is not None and os.path.realpath(marker_base) != os.path.realpath(self.base_dir):
            raise PullError(f"{local} was built from base {marker_base}, this shim's base is {self.base_dir}")
        if applied is not None and target == applied:
            self.stats["noops"] += 1
            return f"already at version {target}"
        if applied is not None and 0 < target < applied:
            raise PullError(f"target_version {target} is behind the local checkpoint (version {applied})")

        # which seed, which deltas
        if target == 0:
            seed, start = self.base_dir, 0
        else:
            floor = applied if applied is not None else 0
            start = target
            while start > floor and _is_delta(version_dir(source, start)):
                start -= 1
            if applied is not None and start <= applied:
                seed, start = None, applied   # continue the local chain
            else:
                seed = self.base_dir if start == 0 else version_dir(source, start)
        if seed is not None and not os.path.isdir(seed):
            raise PullError(f"seed directory missing: {seed}")

        parent = os.path.dirname(local)
        os.makedirs(parent, exist_ok=True)
        tmp = os.path.join(parent, f".pull-tmp-{os.path.basename(local)}-{os.getpid()}-{time.time_ns()}")
        os.mkdir(tmp)
        os.chmod(tmp, 0o755)
        try:
            if seed is None:
                shutil.copytree(local, tmp, dirs_exist_ok=True)
            else:
                _copy_files(seed, tmp)
                write_marker(tmp, start, self.base_dir)
            cur = start
            for v in range(start + 1, target + 1):
                apply_delta(tmp, version_dir(source, v), v, cur, self.workers)
                cur = v
                write_marker(tmp, cur, self.base_dir)
                self.stats["applied_versions"] += 1
            self._swap(tmp, local)
        except BaseException:
            shutil.rmtree(tmp, ignore_errors=True)
            raise
        how = "from the base" if seed == self.base_dir else ("from a full version" if seed else "from the local chain")
        return f"at version {target} ({how}, deltas {start + 1}..{target})" if target > start else \
            f"at version {target} ({how})"

    @staticmethod
    def _swap(tmp: str, local: str) -> None:
        old = None
        if os.path.exists(local):
            old = f"{tmp}.old"
            os.rename(local, old)
        try:
            os.rename(tmp, local)
        except BaseException:
            if old is not None:
                os.rename(old, local)
            raise
        if old is not None:
            shutil.rmtree(old, ignore_errors=True)


# ---------------------------------------------------------------------------
# the proxy
# ---------------------------------------------------------------------------
def map_path(path: str, src_root: str, dst_root: str):
    """path moved from below src_root to below dst_root; None when it is not below src_root."""
    p, s = path.replace("\\", "/"), src_root.replace("\\", "/").rstrip("/")
    if p == s or p.startswith(s + "/"):
        return dst_root.rstrip("/") + p[len(s):]
    return None


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "pull-weights-shim"

    def log_message(self, fmt, *a):
        if getattr(self.server, "verbose", False):
            sys.stderr.write("shim: " + (fmt % a) + "\n")

    # ---- body ----
    def _read_body(self) -> bytes:
        if "chunked" in self.headers.get("Transfer-Encoding", "").lower():
            out = bytearray()
            while True:
                size = int(self.rfile.readline().split(b";")[0].strip() or b"0", 16)
                if size == 0:
                    while self.rfile.readline().strip():   # trailers
                        pass
                    return bytes(out)
                out += self.rfile.read(size)
                self.rfile.readline()
        n = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(n) if n else b""

    def _answer(self, code: int, obj) -> None:
        raw = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    # ---- /pull_weights ----
    def _pull_weights(self, body: bytes) -> None:
        try:
            req = json.loads(body or b"{}")
        except ValueError:
            return self._answer(422, {"detail": [{"type": "json_invalid", "loc": ["body"], "msg": "invalid JSON"}]})
        bad = []
        if not isinstance(req, dict):
            bad = ["body"]
        else:
            for field, typ in (("local_checkpoint_dir", str), ("source_dir", str), ("target_version", int)):
                v = req.get(field)
                if not isinstance(v, typ) or isinstance(v, bool):
                    bad.append(field)
        if bad:
            return self._answer(422, {"detail": [{"type": "invalid", "loc": ["body", f], "msg": "missing or wrong type"}
                                                 for f in bad]})
        t0 = time.time()
        try:
            msg = self.server.puller.pull(req["local_checkpoint_dir"], req["source_dir"], req["target_version"])
        except PullError as e:
            self.server.log(f"pull_weights target={req['target_version']} REFUSED: {e}")
            return self._answer(400, {"success": False, "message": str(e)})
        self.server.log(f"pull_weights target={req['target_version']} ok in {time.time() - t0:.2f}s: {msg}")
        self._answer(200, {"success": True, "message": "Success."})

    # ---- everything else ----
    def _rewrite(self, body: bytes) -> bytes:
        s = self.server
        if not (s.host_root and self.command == "POST"
                and urllib.parse.urlsplit(self.path).path == "/update_weights_from_disk"):
            return body
        try:
            obj = json.loads(body)
        except ValueError:
            return body
        if isinstance(obj, dict) and isinstance(obj.get("model_path"), str):
            new = map_path(obj["model_path"], s.host_root, s.served_root)
            if new is not None:
                obj["model_path"] = new
                return json.dumps(obj).encode()
        return body

    def _forward(self, body: bytes) -> None:
        s = self.server
        body = self._rewrite(body)
        conn = http.client.HTTPConnection(s.up_host, s.up_port, timeout=s.upstream_timeout)
        try:
            conn.putrequest(self.command, self.path, skip_host=True, skip_accept_encoding=True)
            sent_host = False
            for k, v in self.headers.items():
                lk = k.lower()
                if lk in HOP_BY_HOP or lk == "content-length":
                    continue
                sent_host = sent_host or lk == "host"
                conn.putheader(k, v)
            if not sent_host:
                conn.putheader("Host", f"{s.up_host}:{s.up_port}")
            if body or self.command in ("POST", "PUT", "PATCH") or "Content-Length" in self.headers:
                conn.putheader("Content-Length", str(len(body)))
            conn.endheaders(body if body else None)
            resp = conn.getresponse()
        except Exception as e:   # noqa: BLE001
            conn.close()
            return self._answer(502, {"detail": f"upstream {s.up_host}:{s.up_port} unreachable: "
                                                f"{type(e).__name__}: {e}"})
        try:
            self._relay(resp)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            self.close_connection = True
        finally:
            conn.close()

    def _relay(self, resp) -> None:
        chunked_up = bool(resp.chunked)
        no_body = self.command == "HEAD" or resp.status in (204, 304) or 100 <= resp.status < 200
        length = resp.getheader("Content-Length")
        self.send_response_only(resp.status, resp.reason)
        for k, v in resp.getheaders():
            if k.lower() in HOP_BY_HOP or (k.lower() == "content-length" and chunked_up):
                continue
            self.send_header(k, v)
        http11 = self.request_version == "HTTP/1.1"
        mode = "none" if no_body else ("length" if length is not None and not chunked_up else
                                       ("chunked" if http11 else "close"))
        if mode == "chunked":
            self.send_header("Transfer-Encoding", "chunked")
        if mode == "close" or (length is None and not chunked_up and not no_body):
            self.send_header("Connection", "close")
            self.close_connection = True
        self.end_headers()
        if mode == "none":
            return
        while True:
            data = resp.read1(CHUNK)
            if not data:
                break
            if mode == "chunked":
                self.wfile.write(b"%x\r\n" % len(data) + data + b"\r\n")
            else:
                self.wfile.write(data)
            self.wfile.flush()
        if mode == "chunked":
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        if mode == "close":
            self.close_connection = True

    def _handle(self) -> None:
        try:
            body = self._read_body()
        except (ValueError, OSError):
            self.close_connection = True
            return self._answer(400, {"detail": "bad request body"})
        if self.command == "POST" and urllib.parse.urlsplit(self.path).path == "/pull_weights":
            return self._pull_weights(body)
        self._forward(body)

    do_GET = do_POST = do_PUT = do_DELETE = do_PATCH = do_HEAD = do_OPTIONS = _handle


class ShimServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, listen, upstream, base_dir, host_root=None, served_root=None, adopt_unmarked=False,
                 upstream_timeout=3600.0, verbose=False):
        if bool(host_root) != bool(served_root):
            raise ValueError("give --host-root and --served-root together or not at all")
        u = urllib.parse.urlsplit(upstream)
        self.up_host, self.up_port = u.hostname, u.port or 80
        self.host_root, self.served_root = host_root, served_root
        self.upstream_timeout, self.verbose = upstream_timeout, verbose
        self.puller = Puller(base_dir, adopt_unmarked)
        super().__init__(listen, Handler)

    def log(self, msg: str) -> None:
        if self.verbose:
            sys.stderr.write(f"shim {time.strftime('%H:%M:%S')} {msg}\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--listen-host", default="127.0.0.1")
    ap.add_argument("--upstream", default="http://127.0.0.1:30000", help="the real server")
    ap.add_argument("--base-dir", required=True, help="host path of the base checkpoint (the server's model path)")
    ap.add_argument("--host-root", help="host folder mounted into the container")
    ap.add_argument("--served-root", help="where the container sees --host-root (for example /models)")
    ap.add_argument("--adopt-unmarked", action="store_true",
                    help="allow replacing a non-empty local checkpoint directory that has no marker")
    ap.add_argument("--upstream-timeout", type=float, default=3600.0)
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    srv = ShimServer((a.listen_host, a.port), a.upstream, a.base_dir, a.host_root, a.served_root, a.adopt_unmarked,
                     a.upstream_timeout, verbose=not a.quiet)
    srv.log(f"listening on {a.listen_host}:{a.port}, upstream {a.upstream}, base {a.base_dir}, "
            f"path map {a.host_root} -> {a.served_root}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
