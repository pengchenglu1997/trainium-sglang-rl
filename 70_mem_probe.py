#!/usr/bin/env python3
"""Device memory of Qwen3 training on one NeuronCore, stage by stage (plan rows W2.1 and W2.2).

    NEURON_RT_VISIBLE_CORES is set from --core before torch_neuronx is imported.
    python 70_mem_probe.py --model DIR --device neuron --core 0 --results DIR --tag w21 \
        --stages load,forward                              # W2.1: 8B load + one forward
    python 70_mem_probe.py --model DIR_0.6B --device neuron --core 0 --results DIR --tag w22 \
        --stages load,forward,lora_ckpt,lora_nockpt [--compile]   # W2.2: checkpoint comparison
    python 70_mem_probe.py --tiny --device cpu --results DIR --tag cpu --seq 40 --max-completion 16

Stages (--stages, comma list, run in this order):
  load        plain model in bf16 on the device (transformers, no LoRA)           -> "after_load"
  forward     one eager no-grad forward at 1 x --seq on that model                -> "after_forward"
  lora_ckpt   trainer8b.Trainer8b (LoRA as trainer.py attaches it), one forward+backward of one
              sequence with per-layer checkpointing                               -> "lora_ckpt"
  lora_nockpt the same without checkpointing (checkpoint wrappers switched off on the same model;
              it runs last because it is the one most likely to run out of memory)  -> "lora_nockpt"
The plain model is freed before the LoRA stages (a "freed" line records what stayed allocated).
--compile puts each decoder layer under torch.compile(backend="neuron") in the LoRA stages (Neuron only).
Each stage is repeated --repeats times (default 2: the first call includes compilation).

One result line per stage: status ok or fail, seconds of every repeat, and device free / total bytes
before and after (torch.neuron.mem_get_info() when present, else torch.cuda.mem_get_info(); on CPU
the process RSS). For the LoRA stages "worst" is the lowest free memory seen inside the step (taken
after the forward and after the backward by trainer8b.mem_note). A failing stage (an out-of-memory
error included) is written with its error text and the probe goes on with the next stage, or stops
at once with --stop-on-fail. A last line carries {"done": true}.

Expected: free memory falls at "after_load" by about the bf16 weight bytes (Qwen3-8B 16.4 GB on a 25.8 GB
core); lora_nockpt uses clearly more than lora_ckpt at the same shape. Memory numbers are what the runtime
reports, never estimates. Not checkable without a Neuron device: whether mem_get_info exists and what it
counts, the real sizes, out-of-memory behaviour and compile effects; on CPU the script only checks that the
stages run and write their lines."""
from __future__ import annotations

import argparse
import gc
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rl_common as C  # noqa: E402

STAGES = ["load", "forward", "lora_ckpt", "lora_nockpt"]


def rss_bytes():
    try:
        import psutil
        return int(psutil.Process(os.getpid()).memory_info().rss)
    except Exception:  # noqa: BLE001
        try:
            with open("/proc/self/status") as fh:
                for line in fh:
                    if line.startswith("VmRSS:"):
                        return int(line.split()[1]) * 1024
        except Exception:  # noqa: BLE001
            pass
    return None


class Mem:
    """Device memory reader: (free, total) from the runtime when the call exists, else process RSS."""

    def __init__(self, torch, device):
        self.torch, self.device = torch, device
        self.fn, self.source = None, "rss"
        if device == "neuron":
            f = getattr(getattr(torch, "neuron", None), "mem_get_info", None)
            if callable(f):
                self.fn, self.source = f, "torch.neuron.mem_get_info"
        elif device == "cuda":
            self.fn, self.source = torch.cuda.mem_get_info, "torch.cuda.mem_get_info"

    def sync(self):
        try:
            if self.device == "neuron":
                import torch_neuronx
                f = getattr(torch_neuronx, "synchronize", None) or self.torch.neuron.synchronize
                f()
            elif self.device == "cuda":
                self.torch.cuda.synchronize()
        except Exception:  # noqa: BLE001
            pass

    def empty_cache(self):
        try:
            if self.device == "neuron":
                import torch_neuronx
                torch_neuronx.empty_cache()
            elif self.device == "cuda":
                self.torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass

    def read(self):
        self.sync()
        if self.fn is not None:
            try:
                free, total = self.fn()
                return {"source": self.source, "free_bytes": int(free), "total_bytes": int(total)}
            except Exception as e:  # noqa: BLE001
                return {"source": self.source, "error": f"{type(e).__name__}: {e}"[:200], "rss_bytes": rss_bytes()}
        return {"source": "rss", "rss_bytes": rss_bytes()}


def flat(prefix, d):
    return {f"{prefix}_{k}": v for k, v in d.items() if k != "source"}


def switch_checkpoint(tr, flag):
    """Re-wrap the decoder layers of a Trainer8b with checkpointing on or off (the wrapper reads the flag
    when it is built, so the old instance-level forward is removed first)."""
    for layer in tr.base.model.layers:
        if "forward" in layer.__dict__:
            del layer.__dict__["forward"]
        layer._t8b_wrapped = False
    tr.checkpoint = bool(flag)
    tr._wrap_layers()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--tiny", action="store_true", help="CPU self-check: build the tiny random Qwen3 of 65_trainer8b_cpu_test.py")
    ap.add_argument("--device", choices=["neuron", "cpu"], required=True)
    ap.add_argument("--core", default=None, help="NeuronCore index, sets NEURON_RT_VISIBLE_CORES")
    ap.add_argument("--results", required=True)
    ap.add_argument("--tag", default="mem")
    ap.add_argument("--seq", type=int, default=576)
    ap.add_argument("--max-completion", type=int, default=256, help="answer tokens of the training row; prompt = seq - this")
    ap.add_argument("--stages", default=",".join(STAGES))
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--logit-chunk", type=int, default=64)
    ap.add_argument("--stop-on-fail", action="store_true")
    a = ap.parse_args()
    stages = [s for s in a.stages.split(",") if s]
    bad = [s for s in stages if s not in STAGES]
    if bad or (a.model is None and not a.tiny):
        ap.error(f"unknown stage {bad} or neither --model nor --tiny given")
    if a.seq <= a.max_completion + 1:
        ap.error("--seq must exceed --max-completion + 1")
    res = C.Results(a.results, f"70_mem_{a.tag}")
    tmp = None
    plain = None
    tr = None
    try:
        import trainer as T
        if a.device == "neuron":
            T.set_neuron_env(a.core, compile_train=a.compile)
        import torch
        if a.device == "neuron":
            import torch_neuronx  # noqa: F401
            torch._dynamo.config.automatic_dynamic_shapes = False
        model_dir = a.model
        if a.tiny:
            import importlib
            tmp = tempfile.TemporaryDirectory()
            importlib.import_module("65_trainer8b_cpu_test").build_dir(tmp.name, tied=False, vocab=128)
            model_dir = tmp.name
        mem = Mem(torch, a.device)
        try:
            raw = repr(mem.fn()) if mem.fn else None
        except Exception as e:  # noqa: BLE001
            raw = f"error {type(e).__name__}: {e}"[:200]
        dev = "neuron" if a.device == "neuron" else "cpu"
        from transformers import AutoConfig
        vocab = int(AutoConfig.from_pretrained(model_dir).vocab_size)
        res.write(event="start", device=a.device, core=a.core, model=str(a.model or "tiny"), seq=a.seq,
                  stages=stages, repeats=a.repeats, compile=a.compile, torch=torch.__version__,
                  mem_source=mem.source, mem_raw_sample=raw, **flat("t0", mem.read()))
        gen = torch.Generator().manual_seed(5)
        ids_cpu = torch.randint(3, vocab, (1, a.seq), generator=gen)

        def run_stage(name, fn):
            """fn() -> dict of extra fields, with "seconds": list. One result line; never raises."""
            before = mem.read()
            t0 = time.time()
            try:
                extra = fn()
                mem.sync()
                after = mem.read()
                res.write(event="stage", stage=name, status="ok", wall_s=round(time.time() - t0, 3),
                          **flat("before", before), **flat("after", after), **extra)
                ok = True
            except Exception as e:  # noqa: BLE001
                msg = f"{type(e).__name__}: {e}"[:700]
                ok = False
                res.write(event="stage", stage=name, status="fail", error=msg, wall_s=round(time.time() - t0, 3),
                          **flat("before", before), **flat("after", mem.read()))
            gc.collect()
            mem.empty_cache()
            return ok

        proceed = True
        if "load" in stages or "forward" in stages:
            def do_load():
                nonlocal plain
                from transformers import AutoModelForCausalLM
                t = time.time()
                plain = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16, low_cpu_mem_usage=True)
                plain.to(dev)
                plain.eval()
                mem.sync()
                return {"seconds": [round(time.time() - t, 2)], "dtype": "bfloat16",
                        "params": sum(p.numel() for p in plain.parameters())}
            proceed = run_stage("load", do_load)
            if not proceed and a.stop_on_fail:
                raise SystemExit("load failed")
        if "forward" in stages and plain is not None:
            def do_forward():
                secs = []
                ids = ids_cpu.to(dev)
                att = torch.ones_like(ids)
                for _ in range(a.repeats):
                    t = time.time()
                    with torch.no_grad():
                        logits = plain(input_ids=ids, attention_mask=att, use_cache=False).logits
                        s = float(logits.float().sum().cpu())      # host read: the device work is done
                    mem.sync()
                    secs.append(round(time.time() - t, 3))
                    del logits
                return {"seconds": secs, "logits_sum": round(s, 3), "batch": 1, "seq": a.seq}
            if not run_stage("forward", do_forward) and a.stop_on_fail:
                raise SystemExit("forward failed")
        elif "forward" in stages:
            res.write(event="stage", stage="forward", status="skipped", reason="no plain model (load failed)")

        if plain is not None:                      # free the plain model before the LoRA stages
            plain = None
            gc.collect()
            mem.empty_cache()
            res.write(event="freed", **flat("now", mem.read()))

        lora = [s for s in ("lora_ckpt", "lora_nockpt") if s in stages]
        if lora:
            def do_trainer():
                nonlocal tr
                import trainer8b as T8
                t = time.time()
                tr = T8.Trainer8b(model_dir, device=a.device, micro_bs=1, checkpoint=True,
                                  compile_layers=a.compile, logit_chunk=a.logit_chunk,
                                  max_prompt=a.seq - a.max_completion, max_completion=a.max_completion,
                                  lr=1e-5, lora_r=16, lora_alpha=32, seed=1)
                mem.sync()
                return {"seconds": [round(time.time() - t, 2)], "trainable": sum(p.numel() for p in tr.params),
                        "trainer_dtype": str(next(tr.base.parameters()).dtype), "lora_r": 16,
                        "trainer_mem_after_load": tr.mem.get("after_load")}
            ok = run_stage("lora_load", do_trainer)
            if not ok and a.stop_on_fail:
                raise SystemExit("lora_load failed")
            if tr is not None:
                n_p = a.seq - a.max_completion
                row = ((ids_cpu[0].tolist(), n_p, a.max_completion), 1.0)

                def make_step(flag):
                    def step():
                        if tr.checkpoint != flag:
                            switch_checkpoint(tr, flag)
                        secs, losses = [], []
                        tr.mem = {}
                        for _ in range(a.repeats):
                            tr.opt.zero_grad(set_to_none=True)
                            t = time.time()
                            loss, _lp, _m = tr.accumulate([row], 1.0, 1)
                            mem.sync()
                            secs.append(round(time.time() - t, 3))
                            losses.append(round(loss, 5))
                        tr.opt.zero_grad(set_to_none=True)
                        worst = {k: {x: y for x, y in v.items() if x != "stage"} for k, v in tr.mem.items()}
                        return {"seconds": secs, "loss": losses, "checkpoint": flag, "compile": a.compile,
                                "batch": 1, "seq": a.seq, "worst": worst}
                    return step
                for name, flag in (("lora_ckpt", True), ("lora_nockpt", False)):
                    if name in stages:
                        if not run_stage(name, make_step(flag)) and a.stop_on_fail:
                            break
    except SystemExit as e:
        res.write(event="stopped", reason=str(e))
    except BaseException as e:  # noqa: BLE001
        res.fail("mem_probe", e)
        if isinstance(e, KeyboardInterrupt):
            raise
    finally:
        res.done()
        if tmp is not None:
            try:
                tmp.cleanup()
            except Exception:  # noqa: BLE001
                pass


if __name__ == "__main__":
    main()
