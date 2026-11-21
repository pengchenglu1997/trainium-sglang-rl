#!/usr/bin/env python3
"""Can the weights of a model that is already running on a NeuronCore be replaced in place,
without a new process, a new compile or a new warm-up?

    NEURON_RT_VISIBLE_CORES=24 python 45_inplace_swap_probe.py --model DIR --results DIR --tag swap \
        [--device neuron] [--modes eager compiled] [--seq 128] [--batch 8]

For each mode (eager; torch.compile with the Neuron backend) one fixed-shape forward is run until
it is warm, then every parameter is overwritten in place from CPU copies (param.data.copy_), the
same forward is run again, and the original weights are copied back. Recorded: seconds of the
first and of a warm forward, seconds of the swap, seconds of the first and second forward after
the swap, whether the logits changed after the swap, and whether they returned exactly after
swapping back. A first forward after the swap that costs about a warm forward means the running
graphs read the weights as data; one that costs about the very first forward means a recompile.
The new weights are the old ones plus small seeded noise on every matrix: the question is the
mechanism and its time, not what the model then says."""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rl_common as C  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--device", default="neuron", choices=["neuron", "cuda", "cpu"])
    ap.add_argument("--modes", nargs="+", default=["eager", "compiled"])
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--noise", type=float, default=0.01)
    a = ap.parse_args()
    res = C.Results(a.results, f"45_swap_{a.tag}")
    try:
        import torch
        tn = None
        if a.device == "neuron":
            import torch_neuronx as tn  # noqa: F401
            torch._dynamo.config.automatic_dynamic_shapes = False
        from transformers import AutoModelForCausalLM, AutoTokenizer
        dev = {"neuron": "neuron", "cuda": "cuda:0", "cpu": "cpu"}[a.device]

        def sync():
            if a.device == "cuda":
                torch.cuda.synchronize()
            elif tn is not None:
                tn.synchronize()

        tok = AutoTokenizer.from_pretrained(a.model)
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        dtype = torch.float32 if a.device == "cpu" else torch.bfloat16
        t0 = time.time()
        model = AutoModelForCausalLM.from_pretrained(a.model, dtype=dtype, low_cpu_mem_usage=True)
        orig = [p.detach().clone() for p in model.parameters()]          # CPU copies
        g = torch.Generator().manual_seed(7)
        new = [(w.float() + a.noise * w.float().std() * torch.randn(w.shape, generator=g)).to(w.dtype)
               if w.dim() >= 2 else w.clone() for w in orig]
        n_params = sum(w.numel() for w in orig)
        model.to(dev).eval()
        sync()
        res.write(event="loaded", device=a.device, params=n_params, tensors=len(orig),
                  bytes=sum(w.numel() * w.element_size() for w in orig), seconds=round(time.time() - t0, 2),
                  env_compile=os.environ.get("TORCH_COMPILE"), torch=torch.__version__)
        enc = tok(["A farmer has 17 rows of 23 apple trees. How many trees is that?"] * a.batch,
                  return_tensors="pt", padding="max_length", max_length=a.seq, truncation=True)
        ids, att = enc["input_ids"].to(dev), enc["attention_mask"].to(dev)

        def mem(peak=False):      # bytes held on the device, None when the runtime has no counter
            mod = torch.cuda if a.device == "cuda" else tn
            f = getattr(mod, "max_memory_allocated" if peak else "memory_allocated", None)
            try:
                return int(f()) if callable(f) else None
            except Exception:  # noqa: BLE001
                return None

        def swap(src):
            t = time.time()
            with torch.no_grad():
                for p, w in zip(model.parameters(), src):
                    p.data.copy_(w)
            sync()
            return round(time.time() - t, 3)

        for mode in a.modes:
            try:
                fn = model
                if mode == "compiled":
                    fn = torch.compile(model, backend="neuron") if a.device == "neuron" else torch.compile(model)

                def fwd():
                    t = time.time()
                    with torch.no_grad():
                        out = fn(input_ids=ids, attention_mask=att).logits[:, -1, :].float().cpu()
                    sync()
                    return out, round(time.time() - t, 3)

                base, first_s = fwd()
                warm = [fwd()[1] for _ in range(3)]
                mem_before = mem()
                swap_s = swap(new)
                mem_after = mem()
                after, after_first_s = fwd()
                _, after_second_s = fwd()
                back_s = swap(orig)
                again, back_first_s = fwd()
                res.write(event="mode", mode=mode, batch=a.batch, seq=a.seq,
                          first_forward_s=first_s, warm_forward_s=warm,
                          swap_s=swap_s, forward_after_swap_s=after_first_s,
                          second_forward_after_swap_s=after_second_s,
                          swap_back_s=back_s, forward_after_swap_back_s=back_first_s,
                          device_bytes_before_swap=mem_before, device_bytes_after_swap=mem_after,
                          device_peak_bytes_end=mem(peak=True),
                          logits_changed_max_abs=round(float((after - base).abs().max()), 5),
                          logits_after_swap_back_max_abs_diff=round(float((again - base).abs().max()), 7),
                          argmax_changed_rows=int((after.argmax(-1) != base.argmax(-1)).sum()))
            except BaseException as e:  # noqa: BLE001
                res.write(event="mode", mode=mode, error=f"{type(e).__name__}: {e}"[:600])
                if isinstance(e, KeyboardInterrupt):
                    raise
    except BaseException as e:  # noqa: BLE001
        res.fail("swap", e)
        raise
    finally:
        res.done()


if __name__ == "__main__":
    main()
