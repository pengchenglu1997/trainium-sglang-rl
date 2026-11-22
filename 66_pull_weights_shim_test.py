#!/usr/bin/env python3
"""Test of 66_pull_weights_shim.py with Miles's own delta publisher. Prints PASS or FAILED per check and a count.

    <python with torch, safetensors, numpy, zstandard, xxhash, blake3> 66_pull_weights_shim_test.py [--miles-dir DIR]

--miles-dir holds the Miles files read on 22 November (radixark/miles, main at 9011e81): miles/utils/disk_delta.py,
delta.py (miles/backends/training_utils/weight_update/protocols/delta.py), protocol.py, session.py, and
tests/fast/backends/training_utils/weight_update/test_disk_delta_engine_calls.py is the model for how the publisher is
driven. Default: the miles_dl folder of the session scratchpad.

What is Miles's own code: disk_delta.py, protocol.py, session.py and delta.py (class UpdateWeightFromDiskDelta) are
imported by path with importlib and run unchanged. The publisher is driven as Miles's own unit test drives it
(begin_sync with the baseline capture from a real --hf-checkpoint directory, send_bucket per tensor, after_base_weights,
finalize), so the weight_vNNNNNN/ directories are written by Miles's code: xor with xxh3-128, overwrite with blake3, xor
with adler32. What is stubbed, because it cannot be imported without installing Miles and its GPU stack:
  * torch.distributed as used inside delta.py (patched with a MagicMock: rank 0, world size 1, all_gather_object copies
    the local value in, barrier does nothing), get_gloo_group, and UpdateWeightFromDiskDelta._record_metrics (needs
    CUDA); the same patches Miles's own test uses;
  * these modules are replaced by empty stand-ins: miles.backends.sglang_utils.sglang_api_client (SGLangApiClient),
    miles.backends.training_utils.parallel (ParallelState), ...weight_update.hf_weight_iterator (WeightUpdatePlacement),
    ...protocols.utils.data_replica, ...weight_update.rollout_cell_updater, miles.utils.async_utils (submit and
    wait_futures, never called because the rollout engine list is empty), miles.utils.args.custom_view,
    miles.utils.distributed_utils, miles.utils.arguments (driver_owns_generation_pause);
  * rollout_engines of the publisher is an empty list, so Miles's own HTTP client is not used. The HTTP calls come
    from 64_miles_replay.py, our replay of Miles's sequence, run once per published version (with the baseline pull
    on the first), against the shim.
The shim runs in a thread in this process. Its upstream is the stand-in server of selftest_fake_server.py (class H),
extended in this file with a few routes (echo, binary, chunked, status) and a served-path check for
/update_weights_from_disk.

Checks, per encoding (xor/xxh3-128, overwrite/blake3, xor/adler32): after each of three versions with known changes
(plus an unchanged fourth in the first) the replay is accepted by Miles's acceptance checks as our replay models them
(64_miles_replay.miles_accepts, a re-implementation, not Miles's code), the files of
local_checkpoint_dir are bit-identical to the trainer-side weights of that version (safetensors written again from the
trainer tensors) and the marker is right; /pull_weights never reaches the upstream; the update_weights_from_disk body
reaches the upstream with the served path. Then, once: a repeated pull is a no-op, a version missing from the chain is
refused, a corrupted delta (payload byte, checksum, truncation, base_version) is refused and leaves the directory and
its marker as they were with no temporary directory left, catch-up over several versions, reset by target_version 0,
refusal of a target behind the marker, of an unmarked non-empty directory and of the base as target, a 422 for a bad
body, and the pass-through of other routes byte for byte, including a chunked response that streams, a chunked request,
HEAD, an error status, and a 502 when the upstream is down.
Expected result: every line PASS, then "66_pull_weights_shim_test passed (N of N)"."""
from __future__ import annotations

import argparse
import hashlib
import http.client
import importlib.util
import io
import json
import os
import random
import shutil
import sys
import tempfile
import threading
import time
import types
from argparse import Namespace
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest.mock import MagicMock, patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rl_common as C  # noqa: E402
import selftest_fake_server as fake  # noqa: E402

DEFAULT_MILES = os.environ.get("MILES_DIR", str(HERE / "miles_dl"))  # a checkout of radixark/miles
SERVED = "/models/t"
OUT: list = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAILED ") + name + (f"   [{detail}]" if detail and not cond else ""), flush=True)
    OUT.append(bool(cond))


def load_path(modname, path):
    spec = importlib.util.spec_from_file_location(modname, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Miles's modules
# ---------------------------------------------------------------------------
def stub(name, **attrs):
    m = types.ModuleType(name)
    m.__path__ = []
    m.__dict__.update(attrs)
    sys.modules[name] = m
    parent, _, leaf = name.rpartition(".")
    if parent in sys.modules:
        setattr(sys.modules[parent], leaf, m)
    return m


def load_miles(miles_dir: Path):
    import torch  # noqa: F401  (delta.py imports it)
    from dataclasses import dataclass

    @dataclass
    class Placement:
        gather_pp: bool = True

    for pkg in ("miles", "miles.backends", "miles.backends.sglang_utils", "miles.backends.training_utils",
                "miles.backends.training_utils.weight_update", "miles.backends.training_utils.weight_update.protocols",
                "miles.backends.training_utils.weight_update.protocols.utils", "miles.utils", "miles.utils.args"):
        stub(pkg)
    stub("miles.backends.sglang_utils.sglang_api_client", SGLangApiClient=type("SGLangApiClient", (), {}))
    stub("miles.backends.training_utils.parallel", ParallelState=type("ParallelState", (), {}))
    stub("miles.backends.training_utils.weight_update.hf_weight_iterator", WeightUpdatePlacement=Placement)
    stub("miles.backends.training_utils.weight_update.protocols.utils.data_replica",
         get_data_replica_rank_and_size=lambda *a, **k: (0, 1))
    stub("miles.backends.training_utils.weight_update.rollout_cell_updater",
         create_rollout_cell_updaters=lambda *a, **k: {}, _RolloutCellUpdater=type("_RolloutCellUpdater", (), {}))
    pool = ThreadPoolExecutor(2)
    import asyncio
    stub("miles.utils.async_utils", submit=lambda coro: pool.submit(asyncio.run, coro),
         wait_futures=lambda futs: [f.result() for f in futs])
    stub("miles.utils.args.custom_view", compute_custom_function_config=lambda *a, **k: None)
    stub("miles.utils.distributed_utils", get_gloo_group=lambda: None)
    stub("miles.utils.arguments", driver_owns_generation_pause=lambda args: False)
    load_path("miles.utils.disk_delta", miles_dir / "miles/utils/disk_delta.py")
    wu = "miles.backends.training_utils.weight_update"
    load_path(f"{wu}.protocol", miles_dir / "protocol.py")
    load_path(f"{wu}.session", miles_dir / "session.py")
    return load_path(f"{wu}.protocols.delta", miles_dir / "delta.py")


# ---------------------------------------------------------------------------
# trainer-side weights
# ---------------------------------------------------------------------------
SHARDS = {"model-00001-of-00002.safetensors": ["model.embed.weight", "model.layers.0.w"],
          "model-00002-of-00002.safetensors": ["model.layers.1.w", "lm_head.weight"]}


def make_states(torch, n_versions):
    """versions[0] is the base checkpoint, versions[k] the trainer weights after update k (known changes)."""
    g = torch.Generator().manual_seed(1234)
    v0 = {"model.embed.weight": torch.randn(16, 8, generator=g),
          "model.layers.0.w": torch.randn(8, 8, generator=g).to(torch.bfloat16),
          "model.layers.1.w": torch.randn(8, 8, generator=g).to(torch.float16),
          "lm_head.weight": torch.randn(16, 8, generator=g)}
    states = [v0]

    def nxt(edit):
        d = {k: t.clone() for k, t in states[-1].items()}
        edit(d)
        states.append(d)

    def e1(d):
        d["model.embed.weight"][3, :] += 1.0                                    # one row
        d["model.layers.0.w"] = torch.randn(8, 8, generator=g).to(torch.bfloat16)  # every element

    def e2(d):
        d["model.layers.1.w"][2, 5] += 0.5                                      # one element
        d["model.embed.weight"][3, 0] -= 2.0                                    # same row again

    def e3(d):
        d["lm_head.weight"][0, 0] = 7.0                                         # one element

    for e in (e1, e2, e3)[:min(n_versions, 3)]:
        nxt(e)
    if n_versions == 4:
        nxt(lambda d: None)                                                     # nothing changes
    return states


def shard_bytes(st, state):
    return {fn: st.torch.save({n: state[n] for n in names}, metadata={"format": "pt"})
            for fn, names in SHARDS.items()}


def write_base(st, d: Path, state):
    d.mkdir(parents=True, exist_ok=True)
    for fn, blob in shard_bytes(st, state).items():
        (d / fn).write_bytes(blob)
    wm = {n: fn for fn, names in SHARDS.items() for n in names}
    (d / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": wm}))
    (d / "config.json").write_text(json.dumps({"architectures": ["Fake"], "hidden_size": 8}))
    (d / "tokenizer.json").write_bytes(b"{\"fake\": true}\n\x00\xff")


def dir_matches(local: Path, expect: dict, base: Path) -> bool:
    """Every file of local is the expected bytes (shards) or the base's bytes (the rest); only .weight_sync is extra."""
    names = {p.name for p in local.iterdir() if p.is_file()}
    want = {p.name for p in base.iterdir() if p.is_file()}
    if names != want or {p.name for p in local.iterdir() if p.is_dir()} != {".weight_sync"}:
        return False
    for n in want:
        exp = expect[n] if n in expect else (base / n).read_bytes()
        if (local / n).read_bytes() != exp:
            return False
    return True


def snapshot(d: Path):
    return {str(p.relative_to(d)): p.read_bytes() for p in sorted(d.rglob("*")) if p.is_file()}


def marker(local: Path):
    try:
        return json.loads((local / ".weight_sync" / "state.json").read_text())["version"]
    except (OSError, ValueError, KeyError):
        return None


# ---------------------------------------------------------------------------
# servers
# ---------------------------------------------------------------------------
BIN = random.Random(7).randbytes(150_000) + bytes(range(256)) + b"\r\n0\r\n\r\n"
CHUNKS = [b"alpha-" * 50, b"\r\nbeta\r\n", bytes(range(256)) * 20, b"omega"]


class Upstream(fake.H):
    """The stand-in server plus pass-through probes and a served-path check on /update_weights_from_disk."""
    protocol_version = "HTTP/1.1"

    def _json(self, code, obj):
        self._send(code, obj)

    def do_GET(self):
        p = self.path.split("?")[0]
        self.server.state.calls.append(("GET", p))
        if p == "/echo":
            return self._json(200, {"method": "GET", "target": self.path, "probe": self.headers.get("X-Probe"),
                                    "sha": hashlib.sha256(b"").hexdigest()})
        if p == "/binary":
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("X-Up", "1")
            self.send_header("Content-Length", str(len(BIN)))
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(BIN)
            return
        if p == "/chunked":
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Transfer-Encoding", "chunked")
            self.send_header("X-Up", "chunked")
            self.end_headers()
            for i, c in enumerate(CHUNKS):
                self.wfile.write(b"%x\r\n" % len(c) + c + b"\r\n")
                self.wfile.flush()
                if i == 0:
                    time.sleep(0.6)
            self.wfile.write(b"0\r\n\r\n")
            return
        if p == "/status418":
            raw = b"teapot body"
            self.send_response(418, "I'm a teapot")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
            return
        self.server.state.calls.pop()
        return super().do_GET()

    do_HEAD = do_GET

    def do_POST(self):
        n = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(n) if n else b""
        p = self.path.split("?")[0]
        self.server.state.raw.append((self.path, raw))
        if p == "/echo":
            self.server.state.calls.append(("POST", p))
            return self._json(200, {"method": "POST", "target": self.path, "probe": self.headers.get("X-Probe"),
                                    "sha": hashlib.sha256(raw).hexdigest(), "len": len(raw),
                                    "ctype": self.headers.get("Content-Type")})
        self.rfile = io.BytesIO(raw)
        return super().do_POST()

    def _miles_route(self, path, body):
        if path == "/update_weights_from_disk":
            mp = body.get("model_path")
            self.server.state.disk_paths.append(mp)
            if isinstance(mp, str):
                if not mp.startswith(SERVED):
                    return self._send(400, {"success": False, "num_paused_requests": 0,
                                            "message": f"{mp} is not under the mount {SERVED}"})
                body = dict(body, model_path=self.server.host_root + mp[len(SERVED):])
        return super()._miles_route(path, body)


def start(server):
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{server.server_address[1]}"


def http_req(base, method, path, body=None, headers=None, encode_chunked=False):
    u = base.split("//")[1]
    host, port = u.split(":")
    c = http.client.HTTPConnection(host, int(port), timeout=30)
    c.request(method, path, body=body, headers=headers or {}, encode_chunked=encode_chunked)
    r = c.getresponse()
    data = r.read()
    out = (r.status, r.reason, dict((k.lower(), v) for k, v in r.getheaders()), data, r.chunked)
    c.close()
    return out


def pull(shim, local, source, target):
    st, _, _, data, _ = http_req(shim, "POST", "/pull_weights", json.dumps(
        {"local_checkpoint_dir": local.as_posix(), "source_dir": source.as_posix(), "target_version": target}),
        {"Content-Type": "application/json"})
    try:
        return st, json.loads(data)
    except ValueError:
        return st, data


# ---------------------------------------------------------------------------
# scenario: publish with Miles's code, replay against the shim
# ---------------------------------------------------------------------------
def publish_and_replay(label, delta_mod, st, root, shim_url, shim, upstream, replay_mod, encoding, algo, n_versions):
    import torch
    states = make_states(torch, n_versions)
    base = root / f"{label}_hf"
    write_base(st, base, states[0])
    delta_dir = root / f"{label}_delta"
    local = root / f"{label}_local_ckpt"
    args = Namespace(hf_checkpoint=str(base), update_weight_disk_dir=str(delta_dir),
                     update_weight_local_checkpoint_dir=local.as_posix(), update_weight_delta_encoding=encoding,
                     update_weight_delta_checksum=algo, custom_update_weight_post_write_path=None,
                     check_weight_update_equal=False, pause_generation_mode="retract")
    proto = delta_mod.UpdateWeightFromDiskDelta(args)
    proto.rollout_engines, proto.is_sender = [], True
    res_dir = root / f"{label}_results"
    # one shim per label: its base is this label's base checkpoint
    shim_srv = shim(base)
    surl = start(shim_srv)
    upstream.state.disk_paths.clear()
    ok_all = True
    for k in range(1, n_versions + 1):
        def buckets(materialize, k=k):
            assert materialize
            for n, t in states[k].items():
                yield [(n, t)]
        with ExitStack() as es:
            dist = es.enter_context(patch.object(delta_mod, "dist"))
            es.enter_context(patch.object(delta_mod, "get_gloo_group", return_value=None))
            es.enter_context(patch.object(proto, "_record_metrics"))
            dist.get_rank.return_value = 0
            dist.get_world_size.return_value = 1
            dist.all_gather_object.side_effect = lambda values, value, **kw: values.__setitem__(0, value)
            assert proto.begin_sync(k, buckets) is True
            proto._use_pinned = False   # CPU test, as in Miles's own test
            for b in buckets(True):
                proto.send_bucket(b)
            proto.after_base_weights()
            proto.finalize(k)
        vdir = delta_dir / f"weight_v{k:06d}"
        index = json.loads((vdir / "model.safetensors.index.json").read_text())
        check(f"[{label}] v{k} written by Miles's publisher: index says {encoding}/{algo}, base {k - 1:06d}",
              index["metadata"]["delta_encoding"] == encoding and index["metadata"]["checksum_format"] == algo
              and index["metadata"]["base_version"] == f"{k - 1:06d}")
        pulls_before = shim_srv.puller.stats["pulls"]
        up_calls_before = len(upstream.state.calls)
        rows = replay_mod.replay(surl, local.as_posix(), C.Results(res_dir, f"v{k}"), source_dir=delta_dir.as_posix(),
                                 target_version=k, weight_version=k, pause_mode="retract", baseline=(k == 1),
                                 verify=True, timeout=60, flush_tries=3)
        summary_ok = rows and all(r["miles_accepts"] for r in rows)
        check(f"[{label}] v{k} replay: every call accepted by Miles's checks as 64_miles_replay.miles_accepts "
              f"models them ({len(rows)} calls)", summary_ok,
              str([(r["name"], r["status"], r["detail"]) for r in rows if not r["miles_accepts"]])[:300])
        check(f"[{label}] v{k} files of local_checkpoint_dir are bit-identical to the trainer weights",
              dir_matches(local, shard_bytes(st, states[k]), base) and marker(local) == f"{k:06d}")
        names = [n for _, n in upstream.state.calls[up_calls_before:]]
        check(f"[{label}] v{k} /pull_weights was answered by the shim, the rest reached the upstream in order",
              "/pull_weights" not in names and [n for n in names if n != "/ready"][:5] ==
              ["/pause_generation", "/flush_cache", "/update_weights_from_disk", "/continue_generation", "/model_info"],
              str(names))
        check(f"[{label}] v{k} update_weights_from_disk reached the upstream with the served path",
              upstream.state.disk_paths[-1] == SERVED + "/" + local.name, str(upstream.state.disk_paths[-1:]))
        want_pulls = pulls_before + (2 if k == 1 else 1)
        check(f"[{label}] v{k} shim counted {want_pulls - pulls_before} pull(s)", shim_srv.puller.stats["pulls"] == want_pulls)
    return dict(states=states, base=base, delta=delta_dir, local=local, srv=shim_srv, url=surl)


# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--miles-dir", default=DEFAULT_MILES)
    a = ap.parse_args(argv)
    miles_dir = Path(a.miles_dir)
    for rel in ("miles/utils/disk_delta.py", "delta.py", "protocol.py", "session.py"):
        if not (miles_dir / rel).exists():
            print(f"missing {miles_dir / rel}: fetch it with gh api repos/radixark/miles/contents/<path>?ref=9011e81 "
                  "--jq .content | base64 -d", file=sys.stderr)
            sys.exit(2)
    import safetensors.torch as st_torch
    st = types.SimpleNamespace(torch=st_torch)
    delta_mod = load_miles(miles_dir)
    replay_mod = load_path("miles_replay_64", HERE / "64_miles_replay.py")
    shim_mod = load_path("pull_weights_shim_66", HERE / "66_pull_weights_shim.py")

    root = Path(tempfile.mkdtemp(prefix="shimtest_")).resolve()
    root_s = root.as_posix()
    up = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
    up.state = fake.State()
    up.state.disk_paths, up.state.raw = [], []
    up.host_root = root_s
    up_url = start(up)
    servers = [up]

    def shim(base, upstream_url=None):
        s = shim_mod.ShimServer(("127.0.0.1", 0), upstream_url or up_url, str(base), host_root=root_s,
                                served_root=SERVED)
        servers.append(s)
        return s

    try:
        runs = {}
        for label, enc, algo, nv in (("xor-xxh3", "xor", "xxh3-128", 4), ("overwrite-blake3", "overwrite", "blake3", 3),
                                     ("xor-adler32", "xor", "adler32", 3)):
            runs[label] = publish_and_replay(label, delta_mod, st, root, None, shim, up, replay_mod, enc, algo, nv)
        r = runs["xor-xxh3"]
        states, base, delta, srv, surl = r["states"], r["base"], r["delta"], r["srv"], r["url"]
        local = r["local"]
        v4_idx = json.loads((delta / "weight_v000004" / "model.safetensors.index.json").read_text())
        check("v4 (nothing changed) was published by Miles with an empty weight_map and no model file",
              v4_idx["weight_map"] == {} and not list((delta / "weight_v000004").glob("*.safetensors")))
        check("v1 delta of the first label covers only the changed tensors (embed, layers.0)",
              sorted(json.loads((delta / "weight_v000001" / "model.safetensors.index.json").read_text())["weight_map"])
              == ["model.embed.weight", "model.layers.0.w"])

        # ---- repeated pull is a no-op
        exp4 = shard_bytes(st, states[4])
        shard = local / "model-00001-of-00002.safetensors"
        m0, n0, pulls0 = shard.stat().st_mtime_ns, srv.puller.stats["noops"], srv.puller.stats["pulls"]
        s1, b1 = pull(surl, local, delta, 4)
        s2, b2 = pull(surl, local, delta, 0 + 4)
        check("a repeated pull of the current version answers 200 success", s1 == 200 and b1 == {"success": True, "message": "Success."}
              and s2 == 200, str((s1, b1)))
        check("a repeated pull is a no-op (counted, files untouched, same content)",
              srv.puller.stats["noops"] == n0 + 2 and shard.stat().st_mtime_ns == m0
              and dir_matches(local, exp4, base) and marker(local) == "000004")

        # ---- catch-up and reset
        c_local = root / "catchup_ckpt"
        s, b = pull(surl, c_local, delta, 3)
        check("fresh directory, target 3: base then deltas 1, 2, 3 in order, bit-identical to v3",
              s == 200 and dir_matches(c_local, shard_bytes(st, states[3]), base) and marker(c_local) == "000003", str(b))
        d_local = root / "catchup2_ckpt"
        pull(surl, d_local, delta, 1)
        s, b = pull(surl, d_local, delta, 3)
        check("directory at v1, target 3: deltas 2 and 3 are applied in order", s == 200 and
              dir_matches(d_local, shard_bytes(st, states[3]), base) and marker(d_local) == "000003", str(b))
        s, b = pull(surl, d_local, delta, 2)
        check("target behind the marker is refused and the directory is untouched", s == 400 and b["success"] is False
              and marker(d_local) == "000003" and dir_matches(d_local, shard_bytes(st, states[3]), base), str((s, b)))
        s, b = pull(surl, d_local, delta, 0)
        check("target_version 0 on a directory at v3 resets it to the base (marker 0)", s == 200 and
              dir_matches(d_local, {}, base) and marker(d_local) == "000000", str((s, b)))
        s, b = pull(surl, d_local, delta, 1)
        check("... and delta 1 applies on top of the reset base", s == 200 and
              dir_matches(d_local, shard_bytes(st, states[1]), base) and marker(d_local) == "000001", str((s, b)))

        # ---- refusals; each leaves the directory and its marker as they were, and no temporary directory
        def refuse(name, source, target, local_dir, expect_state):
            before = snapshot(local_dir)
            s, b = pull(surl, local_dir, source, target)
            tmp_left = [p.name for p in root.iterdir() if p.name.startswith(".pull-tmp-")]
            check(name, s == 400 and isinstance(b, dict) and b.get("success") is False and snapshot(local_dir) == before
                  and marker(local_dir) == expect_state and not tmp_left, f"{s} {b} {tmp_left}")
            return b

        def fresh_at_v1(tag):
            d = root / f"ref_{tag}"
            s, _ = pull(surl, d, delta, 1)
            assert s == 200
            return d

        bad = root / "delta_gap"
        shutil.copytree(delta, bad)
        shutil.rmtree(bad / "weight_v000002")
        b = refuse("a gap in the chain (v2 missing, target 3 from v1) is refused", bad, 3, fresh_at_v1("gap"), "000001")
        check("... and the message names the missing version", "weight_v000002" in b["message"], b["message"])

        def corrupt_copy(tag):
            d = root / f"delta_{tag}"
            shutil.copytree(delta, d)
            return d

        bad = corrupt_copy("flip")
        f2 = sorted((bad / "weight_v000002").glob("*.safetensors"))[0]
        blob = bytearray(f2.read_bytes())
        blob[-1] ^= 0x55
        f2.write_bytes(bytes(blob))
        refuse("a delta with a flipped payload byte is refused", bad, 2, fresh_at_v1("flip"), "000001")

        bad = corrupt_copy("cksum")
        f2 = sorted((bad / "weight_v000002").glob("*.safetensors"))[0]
        blob = f2.read_bytes()
        hl = int.from_bytes(blob[:8], "little")
        header = json.loads(blob[8:8 + hl])
        name0 = next(k for k in header["__metadata__"])
        cs = header["__metadata__"][name0]
        header["__metadata__"][name0] = ("0" if cs[0] != "0" else "1") + cs[1:]
        new_h = json.dumps(header, separators=(",", ":")).encode()
        new_h += b" " * (hl - len(new_h)) if len(new_h) <= hl else b""
        assert len(new_h) == hl
        f2.write_bytes(blob[:8] + new_h + blob[8 + hl:])
        b = refuse("a delta whose stored checksum is wrong is refused", bad, 2, fresh_at_v1("cksum"), "000001")
        check("... and the message says checksum mismatch", "checksum mismatch" in b["message"], b["message"])

        bad = corrupt_copy("trunc")
        f2 = sorted((bad / "weight_v000002").glob("*.safetensors"))[0]
        f2.write_bytes(f2.read_bytes()[:-40])
        refuse("a truncated delta file is refused", bad, 2, fresh_at_v1("trunc"), "000001")

        bad = corrupt_copy("lineage")
        ip = bad / "weight_v000002" / "model.safetensors.index.json"
        idx = json.loads(ip.read_text())
        idx["metadata"]["base_version"] = "000000"
        ip.write_text(json.dumps(idx))
        b = refuse("a delta whose base_version is not the local version is refused", bad, 2, fresh_at_v1("lineage"), "000001")
        check("... and the message says out-of-order", "out-of-order" in b["message"], b["message"])

        bad = corrupt_copy("badversion")
        ip = bad / "weight_v000002" / "model.safetensors.index.json"
        idx = json.loads(ip.read_text())
        idx["metadata"]["version"] = "000007"
        ip.write_text(json.dumps(idx))
        refuse("a delta that declares another version than its directory is refused", bad, 2,
               fresh_at_v1("badversion"), "000001")

        bad = corrupt_copy("extra")
        shutil.copy(next((bad / "weight_v000002").glob("*.safetensors")), bad / "weight_v000002" / "model-00009-of-00009.safetensors")
        b = refuse("a delta whose files repeat a tensor is refused", bad, 2, fresh_at_v1("extra"), "000001")
        check("... and the message says the tensor appears twice", "appears twice" in b["message"], b["message"])

        bad = corrupt_copy("wmap")
        ip = bad / "weight_v000002" / "model.safetensors.index.json"
        idx = json.loads(ip.read_text())
        idx["weight_map"].pop(sorted(idx["weight_map"])[0])
        ip.write_text(json.dumps(idx))
        b = refuse("a delta file set that differs from the index weight_map is refused", bad, 2, fresh_at_v1("wmap"),
                   "000001")
        check("... and the message says the files do not match the weight_map", "do not match" in b["message"],
              b["message"])

        d = root / "unmarked"
        d.mkdir()
        (d / "precious.txt").write_text("keep")
        s, b = pull(surl, d, delta, 0)
        check("a non-empty directory without a marker is refused and kept", s == 400 and (d / "precious.txt").read_text() == "keep"
              and [p.name for p in d.iterdir()] == ["precious.txt"], str((s, b)))
        s, b = pull(surl, base, delta, 0)
        check("the base directory as local_checkpoint_dir is refused", s == 400 and (base / "config.json").exists(), str((s, b)))
        s, b = pull(surl, root / "never_published", delta, 9)
        check("a target above the published versions is refused (version 5 missing)", s == 400 and
              not (root / "never_published").exists(), str((s, b)))
        stt, _, _, data, _ = http_req(surl, "POST", "/pull_weights", json.dumps({"source_dir": "/x"}),
                                      {"Content-Type": "application/json"})
        check("a body without local_checkpoint_dir and target_version answers 422", stt == 422 and
              b"local_checkpoint_dir" in data and b"target_version" in data, str((stt, data[:120])))
        stt, _, _, data, _ = http_req(surl, "POST", "/pull_weights", b"not json")
        check("a body that is not JSON answers 422", stt == 422)

        # ---- full checkpoint version (the engine's other kind): copied as is
        full = root / "delta_full"
        shutil.copytree(delta, full)
        shutil.copytree(base, full / "weight_v000005", ignore=shutil.ignore_patterns("model.safetensors.index.json"))
        for fn, blob in shard_bytes(st, states[4]).items():
            (full / "weight_v000005" / fn).write_bytes(blob)
        f_local = root / "full_ckpt"
        s, b = pull(surl, f_local, full, 5)
        check("a full version in the chain replaces the directory (marker 5, files of v4)", s == 200 and marker(f_local) == "000005"
              and all((f_local / fn).read_bytes() == blob for fn, blob in shard_bytes(st, states[4]).items()), str((s, b)))

        # ---- pass-through
        probe = bytes(range(256)) * 40 + b"\r\n\r\n"
        stt, _, h, data, _ = http_req(surl, "POST", "/echo?a=1&b=%20x&a=2", probe,
                                      {"Content-Type": "application/octet-stream", "X-Probe": "hello"})
        e = json.loads(data)
        check("POST with query, custom header and a binary body reaches the upstream unchanged",
              stt == 200 and e["target"] == "/echo?a=1&b=%20x&a=2" and e["probe"] == "hello"
              and e["sha"] == hashlib.sha256(probe).hexdigest() and e["ctype"] == "application/octet-stream", str(e))
        stt, _, h, data, _ = http_req(surl, "GET", "/echo?q=%C3%A9", headers={"X-Probe": "g"})
        e = json.loads(data)
        check("GET with a query string is forwarded unchanged", stt == 200 and e["target"] == "/echo?q=%C3%A9" and e["probe"] == "g")
        stt, _, h, data, ch = http_req(surl, "GET", "/binary")
        check("a binary answer with Content-Length comes back byte for byte, headers kept",
              stt == 200 and data == BIN and h.get("content-length") == str(len(BIN)) and h.get("x-up") == "1" and not ch)
        stt, _, h, data, ch = http_req(surl, "HEAD", "/binary")
        check("HEAD: status and Content-Length kept, no body", stt == 200 and data == b"" and h.get("content-length") == str(len(BIN)))
        stt, reason, h, data, _ = http_req(surl, "GET", "/status418")
        check("an error status, reason and body pass through", stt == 418 and reason == "I'm a teapot" and data == b"teapot body")
        stt, _, h, data, _ = http_req(surl, "GET", "/pull_weights")
        check("GET /pull_weights is not the shim's: the upstream answers (404)", stt == 404 and b"Not Found" in data)
        stt, _, h, data, _ = http_req(surl, "GET", "/model_info")
        check("GET /model_info passes through", stt == 200 and "weight_version" in json.loads(data))

        # chunked response: byte for byte, still chunked, and the first chunk arrives before the last is sent
        u = surl.split("//")[1].split(":")
        c = http.client.HTTPConnection(u[0], int(u[1]), timeout=30)
        c.request("GET", "/chunked")
        t0 = time.time()
        r = c.getresponse()
        first = r.read1(1 << 16)
        t_first = time.time() - t0
        rest = r.read()
        t_all = time.time() - t0
        check("a chunked answer comes back byte for byte and still chunked", r.chunked and (first + rest) == b"".join(CHUNKS)
              and r.getheader("x-up") == "chunked" and r.getheader("content-length") is None)
        check("... and is streamed (first piece after %.2f s, whole answer after %.2f s)" % (t_first, t_all),
              t_first < 0.4 and t_all >= 0.55)
        c.close()

        # chunked request body
        def gen():
            yield b"part-one;"
            yield probe
            yield b";end"
        stt, _, h, data, _ = http_req(surl, "POST", "/echo", gen(), {"Content-Type": "application/octet-stream"},
                                      encode_chunked=True)
        e = json.loads(data)
        check("a chunked request body reaches the upstream whole", stt == 200 and
              e["sha"] == hashlib.sha256(b"part-one;" + probe + b";end").hexdigest(), str(e)[:200])

        # path mapping
        raw_other = b'{"model_path":   "/elsewhere/x" ,  "weight_version":"9"}'
        stt, _, _, data, _ = http_req(surl, "POST", "/update_weights_from_disk", raw_other, {"Content-Type": "application/json"})
        got = [raw for p, raw in up.state.raw if p == "/update_weights_from_disk"][-1]
        check("an update_weights_from_disk body outside the host root is forwarded byte for byte", got == raw_other and stt == 400)
        body_in = json.dumps({"model_path": local.as_posix(), "weight_version": "11"}).encode()
        stt, _, _, data, _ = http_req(surl, "POST", "/update_weights_from_disk", body_in, {"Content-Type": "application/json"})
        got = json.loads([raw for p, raw in up.state.raw if p == "/update_weights_from_disk"][-1])
        check("a model_path below the host root is rewritten to the served root (other fields kept)",
              stt == 200 and got == {"model_path": SERVED + "/" + local.name, "weight_version": "11"}, str(got))
        check("a sibling folder with the host root as a name prefix is not rewritten",
              shim_mod.map_path(root_s + "x/m", root_s, SERVED) is None and shim_mod.map_path(root_s, root_s, SERVED) == SERVED)

        # upstream down
        s_down = shim(base, upstream_url="http://127.0.0.1:1")
        stt, _, _, data, _ = http_req(start(s_down), "GET", "/model_info")
        check("upstream unreachable: the shim answers 502", stt == 502, str((stt, data[:100])))
        stt, _ = pull(start(shim(base, upstream_url="http://127.0.0.1:1")), root / "down_ckpt", delta, 1)
        check("... but /pull_weights does not need the upstream", stt == 200)
    finally:
        for s in servers:
            s.shutdown()
        shutil.rmtree(root, ignore_errors=True)
    print("66_pull_weights_shim_test", "passed" if all(OUT) else "FAILED", f"({sum(OUT)} of {len(OUT)})")
    sys.exit(0 if all(OUT) else 1)


if __name__ == "__main__":
    main()
