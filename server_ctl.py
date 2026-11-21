"""Start, wait for and stop a model server described in backends.json.

A backend entry:
  "start":  command list; "{model}", "{port}", "{name}" are replaced; run from the kit folder
  "stop":   optional command list to stop it (docker stop); else the process group started by
            "start" gets SIGTERM, then SIGKILL after stop_grace_s
  "alive":  optional command list that prints "true" while the server lives (a detached container)
  "logs":   optional command list whose output is saved to the log file before a stop
  "detached": true when "start" returns at once (docker run -d)
  "served": the model name the API expects; default the model path
  "served_map_env", "served_map_to": the model path as the server sees it (a mounted folder)
  "weight_update": "sglang_disk" when the server takes SGLang's /update_weights_from_disk
  "ready_timeout_s", "stop_grace_s"
Standard library only. Nothing here launches or changes a machine."""
from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from pathlib import Path

import rollout_client as rc

KIT = Path(__file__).resolve().parent


def load_backend(path, name):
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    if name not in cfg:
        raise KeyError(f"backend {name} not in {path}: {sorted(cfg)}")
    return cfg[name]


class Server:
    def __init__(self, backend: dict, port: int, log_dir, name="rlserver", env=None):
        # env: extra environment of this server's commands (a replica's own cores, for example)
        self.env = dict(os.environ, **{k: str(v) for k, v in (env or {}).items()})
        self.b, self.port, self.name = backend, int(port), name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.proc = None
        self.model = None
        self.log = None
        self.starts = 0

    @property
    def base_url(self):
        return f"http://127.0.0.1:{self.port}"

    def _fill(self, parts):
        return [str(p).format(model=self.model or "", port=self.port, name=self.name)
                for p in parts]

    def model_as_served(self, path=None):
        """A model path (default: the one the server was started on) as the server sees it."""
        m = str(path) if path is not None else (self.model or "")
        env = self.b.get("served_map_env")
        root = os.environ.get(env, "") if env else ""
        if root and m.startswith(root.rstrip("/") + "/"):
            return self.b.get("served_map_to", "/models").rstrip("/") + m[len(root.rstrip("/")):]
        return m

    def served_name(self):
        if "served" in self.b:
            return str(self.b["served"]).format(model=self.model_as_served(), name=self.name)
        return self.model_as_served()

    def alive(self):
        if self.b.get("alive"):
            try:
                out = subprocess.run(self._fill(self.b["alive"]), cwd=KIT, env=self.env, capture_output=True,
                                     text=True, timeout=30).stdout.strip()
            except Exception:  # noqa: BLE001
                return False
            return out == "true"
        return self.proc is not None and self.proc.poll() is None

    def _save_logs(self):
        if self.b.get("logs") and self.log is not None:
            try:
                with open(self.log, "ab") as f:
                    subprocess.run(self._fill(self.b["logs"]), cwd=KIT, env=self.env, stdout=f,
                                   stderr=subprocess.STDOUT, timeout=120)
            except Exception:  # noqa: BLE001
                pass

    def start(self, model_path) -> dict:
        """Start on model_path and wait until it answers. ready_s None = it did not come up."""
        self.model = str(model_path)
        self.starts += 1
        self.log = self.log_dir / f"server_{self.name}_{self.starts:03d}.log"
        t0 = time.time()
        with open(self.log, "ab") as f:
            self.proc = subprocess.Popen(self._fill(self.b["start"]), cwd=KIT, env=self.env, stdout=f,
                                         stderr=subprocess.STDOUT, start_new_session=True)
        if self.b.get("detached"):
            try:
                rcode = self.proc.wait(timeout=600)
            except subprocess.TimeoutExpired:
                rcode = None
            if rcode != 0:
                return {"ready_s": None, "log": str(self.log), "start_no": self.starts,
                        "launcher_exit": rcode}
            time.sleep(3)
        ready = rc.wait_ready(self.base_url, self.served_name(),
                              float(self.b.get("ready_timeout_s", 3600)), alive=self.alive)
        out = {"ready_s": None if ready is None else round(time.time() - t0, 1),
               "log": str(self.log), "start_no": self.starts}
        if ready is None:
            self._save_logs()
        return out

    def stop(self) -> float:
        t0 = time.time()
        grace = float(self.b.get("stop_grace_s", 120))
        self._save_logs()
        if self.b.get("stop"):
            try:
                subprocess.run(self._fill(self.b["stop"]), cwd=KIT, env=self.env, stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL, timeout=grace + 120)
            except Exception:  # noqa: BLE001
                pass
        if self.proc is not None and self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, signal.SIGTERM)
            except (ProcessLookupError, AttributeError, PermissionError):
                self.proc.terminate()
            try:
                self.proc.wait(timeout=grace)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(self.proc.pid, signal.SIGKILL)
                except (ProcessLookupError, AttributeError, PermissionError):
                    self.proc.kill()
                self.proc.wait(timeout=30)
        for _ in range(60):          # the port must be free before the next start
            if rc.get(self.base_url + "/health", timeout=2)[0] == 0:
                break
            time.sleep(1)
        self.proc = None
        return round(time.time() - t0, 1)

    def restart(self, model_path) -> dict:
        stop_s = self.stop()
        out = self.start(model_path)
        out["stop_s"] = stop_s
        return out

    def update_weights_from_disk(self, model_path, timeout=1800) -> dict:
        """SGLang's own call for new weights without a restart (standard SGLang; SGLang-Neuron
        v0.1.0 answers HTTP 400). Returns status, seconds and the start of the body."""
        # self.model stays the path of the start: the API keeps answering under that name
        code, out, sec = rc.post(self.base_url + "/update_weights_from_disk",
                                 {"model_path": self.model_as_served(model_path)}, timeout=timeout)
        text = out if isinstance(out, str) else json.dumps(out)
        ok = code == 200 and (not isinstance(out, dict) or out.get("success", True))
        return {"status": code, "ok": bool(ok), "seconds": round(sec, 2), "body": text[:300]}
