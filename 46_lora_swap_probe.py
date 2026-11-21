#!/usr/bin/env python3
"""Adapter-only weight sync on a running NeuronCore: send only the LoRA tensors, keep the base.

    NEURON_RT_VISIBLE_CORES=24 python 46_lora_swap_probe.py --model DIR --results DIR --tag lora \
        [--device neuron] [--seq 128] [--batch 8] [--rank 16]

Two ways a server could take a new adapter without a restart, each timed on one fixed-shape
compiled forward (torch.compile, Neuron backend), same LoRA layout as trainer.py:
  side:   the adapter stays a separate branch in the graph (base + B A x); a new adapter is an
          in-place overwrite of the A and B tensors only.
  merged: no adapter branch in the graph; the delta scale * (B @ A) is computed on the device
          and added in place to each target weight (and the old one subtracted first).
Recorded for each: bytes sent, seconds of the swap, seconds of the forward before and after,
whether the logits changed, and how exactly the first state comes back (side: copy the first
adapter back; merged: subtract the second delta and add the first again, bf16 rounding shows
here). Also the warm forward of the bare base model, to price the side branch.
The adapters are seeded random tensors: the question is mechanism and time."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rl_common as C  # noqa: E402

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--device", default="neuron", choices=["neuron", "cuda", "cpu"])
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    a = ap.parse_args()
    res = C.Results(a.results, f"46_lora_swap_{a.tag}")
    try:
        import torch
        tn = None
        if a.device == "neuron":
            import torch_neuronx as tn  # noqa: F401
            torch._dynamo.config.automatic_dynamic_shapes = False
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer
        dev = {"neuron": "neuron", "cuda": "cuda:0", "cpu": "cpu"}[a.device]
        dtype = torch.float32 if a.device == "cpu" else torch.bfloat16

        def sync():
            if a.device == "cuda":
                torch.cuda.synchronize()
            elif tn is not None:
                tn.synchronize()

        def mem():
            mod = torch.cuda if a.device == "cuda" else tn
            f = getattr(mod, "memory_allocated", None)
            try:
                return int(f()) if callable(f) else None
            except Exception:  # noqa: BLE001
                return None

        def compiled(m):
            return torch.compile(m, backend="neuron") if a.device == "neuron" else torch.compile(m)

        tok = AutoTokenizer.from_pretrained(a.model)
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        enc = tok(["A farmer has 17 rows of 23 apple trees. How many trees is that?"] * a.batch,
                  return_tensors="pt", padding="max_length", max_length=a.seq, truncation=True)
        ids, att = enc["input_ids"].to(dev), enc["attention_mask"].to(dev)

        def timed(fn):
            t = time.time()
            with torch.no_grad():
                out = fn(input_ids=ids, attention_mask=att).logits[:, -1, :].float().cpu()
            sync()
            return out, round(time.time() - t, 3)

        def warm(fn):
            out, first = timed(fn)
            times = [timed(fn)[1] for _ in range(3)]
            return out, first, times

        # ---- bare base model -------------------------------------------------------------
        base = AutoModelForCausalLM.from_pretrained(a.model, dtype=dtype, low_cpu_mem_usage=True).to(dev).eval()
        sync()
        base_bytes = sum(p.numel() * p.element_size() for p in base.parameters())
        fb = compiled(base)
        base_logits, base_first, base_warm = warm(fb)
        res.write(event="base", device=a.device, base_bytes=base_bytes, first_forward_s=base_first,
                  warm_forward_s=base_warm, device_bytes=mem())

        # ---- two seeded adapters on the CPU ----------------------------------------------
        cfg = LoraConfig(r=a.rank, lora_alpha=a.alpha, lora_dropout=0.0, target_modules=LORA_TARGETS,
                         bias="none", task_type="CAUSAL_LM")
        peft_model = get_peft_model(base, cfg).eval()
        peft_model.to(dev)
        sync()
        lora = [(n, p) for n, p in peft_model.named_parameters() if "lora_" in n]
        g = torch.Generator().manual_seed(11)

        def make():
            return [(0.02 * torch.randn(p.shape, generator=g)).to(dtype) for _, p in lora]
        ad1, ad2 = make(), make()
        adapter_bytes = sum(t.numel() * t.element_size() for t in ad1)

        def put(tensors):
            t = time.time()
            with torch.no_grad():
                for (_, p), w in zip(lora, tensors):
                    p.data.copy_(w)
            sync()
            return round(time.time() - t, 4)

        # ---- side: adapter branch stays in the graph --------------------------------------
        try:
            put(ad1)
            fs = compiled(peft_model)
            l1, side_first, side_warm = warm(fs)
            m0 = mem()
            swap_s = put(ad2)
            l2, after_s = timed(fs)
            _, after2_s = timed(fs)
            back_s = put(ad1)
            l1b, _ = timed(fs)
            res.write(event="side", adapter_tensors=len(lora), adapter_bytes=adapter_bytes,
                      share_of_base_bytes=round(adapter_bytes / base_bytes, 5),
                      first_forward_s=side_first, warm_forward_s=side_warm, swap_s=swap_s,
                      forward_after_swap_s=after_s, second_forward_after_swap_s=after2_s,
                      swap_back_s=back_s, device_bytes_before=m0, device_bytes_after=mem(),
                      logits_changed_max_abs=round(float((l2 - l1).abs().max()), 5),
                      logits_after_swap_back_max_abs_diff=round(float((l1b - l1).abs().max()), 7),
                      differs_from_base_max_abs=round(float((l1 - base_logits).abs().max()), 5))
        except BaseException as e:  # noqa: BLE001
            res.write(event="side", error=f"{type(e).__name__}: {e}"[:600])
            if isinstance(e, KeyboardInterrupt):
                raise

        # ---- merged: delta computed on the device and added in place ------------------------
        try:
            scale = a.alpha / a.rank
            targets = []      # (weight parameter, index of A, index of B)
            names = [n for n, _ in lora]
            mods = dict(peft_model.named_modules())
            for i, n in enumerate(names):
                if ".lora_A." in n:
                    mod_name = n.split(".lora_A.")[0]
                    j = names.index(n.replace(".lora_A.", ".lora_B."))
                    targets.append((mods[mod_name].base_layer.weight, i, j))
            dev_ad = {}

            def to_dev(tensors, key):
                t = time.time()
                dev_ad[key] = [w.to(dev) for w in tensors]
                sync()
                return round(time.time() - t, 4)

            def apply(key, sign):
                t = time.time()
                with torch.no_grad():
                    for w, i, j in targets:
                        w.data.add_((dev_ad[key][j].float() @ dev_ad[key][i].float()).to(w.dtype), alpha=sign * scale)
                sync()
                return round(time.time() - t, 4)

            # get_peft_model wrapped the base modules in place, so every graph of this model now
            # has the side branch. Zero the side adapter: from here on only the merged weights
            # carry an adapter. (The first run of this probe, tag "lora", did not do this and
            # its merged logits counted adapter 1 twice; its timings stand, its logit columns do not.)
            put([torch.zeros_like(w) for w in ad1])
            fb = fs
            zero_side, _ = timed(fb)
            side_zero_vs_base = round(float((zero_side - base_logits).abs().max()), 5)
            send1, send2 = to_dev(ad1, 1), to_dev(ad2, 2)
            add1_first = apply(1, +1)                       # first call compiles the small matmuls
            m1, _ = timed(fb)                               # the graph now reads merged weights
            m0 = mem()
            apply(1, -1)                                    # first subtract compiles its own op
            apply(1, +1)
            t = time.time()
            sub_s = apply(1, -1)
            add_s = apply(2, +1)
            swap_total = round(time.time() - t, 4)
            m2, after_s = timed(fb)
            apply(2, -1)
            apply(1, +1)
            m1b, _ = timed(fb)
            apply(1, -1)
            m_base, _ = timed(fb)
            res.write(event="merged", target_weights=len(targets), send_adapter_s=[send1, send2],
                      first_apply_s=add1_first, subtract_old_s=sub_s, add_new_s=add_s,
                      swap_total_s=swap_total, forward_after_swap_s=after_s,
                      side_zeroed_vs_base_max_abs_diff=side_zero_vs_base,
                      device_bytes_before=m0, device_bytes_after=mem(),
                      logits_changed_max_abs=round(float((m2 - m1).abs().max()), 5),
                      logits_after_swap_back_max_abs_diff=round(float((m1b - m1).abs().max()), 5),
                      base_after_all_subtracted_max_abs_diff=round(float((m_base - base_logits).abs().max()), 5),
                      merged_vs_side_same_adapter_max_abs_diff=(
                          round(float((m1 - l1).abs().max()), 5) if "l1" in dir() else None))
        except BaseException as e:  # noqa: BLE001
            res.write(event="merged", error=f"{type(e).__name__}: {e}"[:600])
            if isinstance(e, KeyboardInterrupt):
                raise
    except BaseException as e:  # noqa: BLE001
        res.fail("lora_swap", e)
        raise
    finally:
        res.done()


if __name__ == "__main__":
    main()
