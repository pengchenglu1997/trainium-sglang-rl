#!/usr/bin/env python3
"""Why did the adapter update computed on the NeuronCore not match the host-merged checkpoint
(50_adapter adapter_dev, 2026-11-20: logprobs off by up to 1.1)? Each step of that path is run
on the host and on the device for real adapter tensors and compared.

    NEURON_RT_VISIBLE_CORES=24 python 53_device_delta_probe.py --base DIR --adapter DIR --results DIR --tag delta
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rl_common as C  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    res = C.Results(a.results, f"53_device_delta_{a.tag}")
    try:
        import torch
        import torch_neuronx as tn
        from safetensors.torch import load_file
        ad = load_file(str(Path(a.adapter) / "adapter_model.safetensors"))
        sd = {}
        for f in sorted(Path(a.base).glob("*.safetensors")):
            sd.update(load_file(str(f)))
        dev = "neuron"
        scale = 2.0

        def stat(x, y):
            d = (x.float() - y.float()).abs()
            return {"max_abs": float(d.max()), "mean_abs": float(d.mean()),
                    "ref_mean_abs": float(y.float().abs().mean()), "differing_share": float((d > 0).float().mean())}

        for proj in ("self_attn.q_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.down_proj"):
            key = f"model.layers.3.{proj}.weight"
            A = ad[f"base_model.model.model.layers.3.{proj}.lora_A.weight"].float()
            B = ad[f"base_model.model.model.layers.3.{proj}.lora_B.weight"].float()
            W = sd[key]
            d_host = scale * (B @ A)
            d_dev_t = scale * (B.to(dev) @ A.to(dev))
            tn.synchronize()
            d_dev = d_dev_t.cpu()
            res.write(event="matmul", proj=proj, shape=list(d_host.shape), device_dtype=str(d_dev_t.dtype),
                      **stat(d_dev, d_host))
            # the sum with the base and the cast, host vs device, from the SAME (host) delta
            new_host = (W.float() + d_host).to(W.dtype)
            new_dev = (W.to(dev).float() + d_host.to(dev)).to(W.dtype)
            tn.synchronize()
            res.write(event="sum_cast_same_delta", proj=proj, base_dtype=str(W.dtype), **stat(new_dev.cpu(), new_host))
            # the whole device path as the patch does it
            full_dev = (W.to(dev).float() + d_dev_t).to(W.dtype)
            tn.synchronize()
            res.write(event="full_device_path", proj=proj, **stat(full_dev.cpu(), new_host),
                      ulp_scale_note="bf16 keeps 8 significant bits")
            # slicing, cat and transpose on the device (what the loaders do)
            h_host = torch.cat([d_host[:64], d_host[128:192]], dim=0).t().contiguous()
            h_dev = torch.cat([d_host.to(dev)[:64], d_host.to(dev)[128:192]], dim=0).t().contiguous()
            tn.synchronize()
            res.write(event="slice_cat_t", proj=proj, **stat(h_dev.cpu(), h_host))
            c_host = d_host[:, 256:512].t().contiguous()
            c_dev = d_host.to(dev)[:, 256:512].t().contiguous()
            tn.synchronize()
            res.write(event="column_slice_t", proj=proj, **stat(c_dev.cpu(), c_host))
    except BaseException as e:  # noqa: BLE001
        res.fail("delta", e)
        raise
    finally:
        res.done()


if __name__ == "__main__":
    main()
