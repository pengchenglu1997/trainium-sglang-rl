#!/usr/bin/env python3
"""Inside the vLLM-Neuron container: does the compile backend that engine uses read weights as data?
The same question as 45_inplace_swap_probe.py, asked of vllm_neuron's own torch.compile backend
(the released engine has no weight update; PR #98 adds one by param.data.copy_).

    docker run --rm --device /dev/neuron0 ... --entrypoint python <vllm image> /kit/54_vllm_backend_probe.py /models/Qwen3-0.6B

Prints one JSON line per step, and says which step failed if one does. Standard library plus what
the image ships."""
import json
import sys
import time
import traceback


def out(**kv):
    print(json.dumps(kv), flush=True)


def main():
    model_dir = sys.argv[1]
    import torch
    info = {"torch": torch.__version__}
    for mod in ("torch_neuronx", "libtorch_neuronx_lite", "vllm_neuron"):
        try:
            m = __import__(mod)
            info[mod] = getattr(m, "__version__", "imported")
        except Exception as e:  # noqa: BLE001
            info[mod] = f"import failed: {type(e).__name__}: {str(e)[:120]}"
    out(event="imports", **info)
    backend = None
    try:
        from vllm_neuron import envs
        for name in ("get_compile_backend_name",):
            f = getattr(envs, name, None)
            if callable(f):
                backend = f()
        out(event="backend", name=backend, registered=[b for b in torch._dynamo.list_backends(None) if "neuron" in b.lower()])
    except Exception as e:  # noqa: BLE001
        out(event="backend", error=f"{type(e).__name__}: {str(e)[:300]}")
    try:
        devs = [d for d in ("neuron", "neuron:0") if _dev_ok(torch, d)]
        out(event="device", usable=devs)
        dev = devs[0]
    except Exception as e:  # noqa: BLE001
        out(event="device", error=f"{type(e).__name__}: {str(e)[:300]}")
        return
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16, low_cpu_mem_usage=True)
    orig = [p.detach().clone() for p in model.parameters()]
    g = torch.Generator().manual_seed(7)
    new = [(w.float() + 0.01 * w.float().std() * torch.randn(w.shape, generator=g)).to(w.dtype) if w.dim() >= 2
           else w.clone() for w in orig]
    model.to(dev).eval()
    enc = tok(["A farmer has 17 rows of 23 apple trees. How many trees is that?"] * 4, return_tensors="pt",
              padding="max_length", max_length=128, truncation=True)
    ids, att = enc["input_ids"].to(dev), enc["attention_mask"].to(dev)

    def swap(src):
        t = time.time()
        with torch.no_grad():
            for p, w in zip(model.parameters(), src):
                p.data.copy_(w)
        return round(time.time() - t, 3)

    for mode, bk in (("eager", None), ("compiled", backend)):
        if mode == "compiled" and not bk:
            out(event="mode", mode=mode, skipped="no backend name")
            continue
        try:
            fn = torch.compile(model, backend=bk) if bk else model

            def fwd():
                t = time.time()
                with torch.no_grad():
                    o = fn(input_ids=ids, attention_mask=att).logits[:, -1, :].float().cpu()
                return o, round(time.time() - t, 3)
            base, first = fwd()
            warm = [fwd()[1] for _ in range(3)]
            s1 = swap(new)
            after, t_after = fwd()
            s2 = swap(orig)
            again, _ = fwd()
            out(event="mode", mode=mode, backend=bk, first_forward_s=first, warm_forward_s=warm, swap_s=s1,
                forward_after_swap_s=t_after, swap_back_s=s2,
                logits_changed_max_abs=round(float((after - base).abs().max()), 5),
                logits_after_swap_back_max_abs_diff=round(float((again - base).abs().max()), 7))
        except Exception as e:  # noqa: BLE001
            out(event="mode", mode=mode, backend=bk, error=f"{type(e).__name__}: {str(e)[:500]}",
                where=traceback.format_exc().strip().splitlines()[-3:][0][:200])
    out(done=True)


def _dev_ok(torch, d):
    try:
        torch.zeros(2).to(d).cpu()
        return True
    except Exception:  # noqa: BLE001
        return False


if __name__ == "__main__":
    main()
