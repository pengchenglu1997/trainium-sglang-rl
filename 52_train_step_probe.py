#!/usr/bin/env python3
"""Where the trainer's time goes now that weight sync is cheap: one LoRA training micro-batch on a
NeuronCore, as the loop runs it (eager) and with the model under torch.compile (Neuron backend).

    NEURON_RT_VISIBLE_CORES=24 python 52_train_step_probe.py --model DIR --results DIR --tag train \
        [--batch 4] [--seq 576] [--modes eager compiled] [--repeats 6]

Per mode: seconds of the first forward + backward (compile included) and of the following ones,
the peak bytes on the device, and the loss of the first call (the two modes should agree on it).
Same LoRA layout, shape and loss form as trainer.py; the tokens are a fixed prompt repeated to the
length, since only the cost is measured."""
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
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seq", type=int, default=576)
    ap.add_argument("--modes", nargs="+", default=["eager", "compiled"])
    ap.add_argument("--repeats", type=int, default=6)
    a = ap.parse_args()
    res = C.Results(a.results, f"52_train_step_{a.tag}")
    try:
        import torch
        import torch_neuronx as tn
        torch._dynamo.config.automatic_dynamic_shapes = False
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(a.model)
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        base = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, low_cpu_mem_usage=True).to("neuron")
        model = get_peft_model(base, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, target_modules=LORA_TARGETS,
                                                bias="none", task_type="CAUSAL_LM"))
        model.train()
        params = [p for p in model.parameters() if p.requires_grad]
        one = tok("A farmer has 17 rows of 23 apple trees. He cuts down 41 trees. How many trees are left? ",
                  add_special_tokens=False)["input_ids"]
        seq = (one * (a.seq // len(one) + 1))[:a.seq]
        ids = torch.tensor([seq] * a.batch, dtype=torch.long).to("neuron")
        att = torch.ones_like(ids)
        tn.synchronize()
        res.write(event="loaded", batch=a.batch, seq=a.seq, trainable=sum(p.numel() for p in params))

        def peak(reset=False):
            f = getattr(tn, "reset_peak_memory_stats" if reset else "max_memory_allocated", None)
            try:
                return f() if callable(f) else None
            except Exception:  # noqa: BLE001
                return None

        for mode in a.modes:
            try:
                fn = torch.compile(model, backend="neuron") if mode == "compiled" else model
                peak(reset=True)
                times, losses = [], []
                for _ in range(a.repeats):
                    t = time.time()
                    for p in params:
                        p.grad = None
                    logits = fn(input_ids=ids, attention_mask=att).logits
                    logp = torch.log_softmax(logits[:, :-1, :].float(), dim=-1)
                    loss = -logp.gather(-1, ids[:, 1:].unsqueeze(-1)).mean()
                    loss.backward()
                    val = float(loss.detach().float().cpu())
                    tn.synchronize()
                    times.append(round(time.time() - t, 3))
                    losses.append(round(val, 5))
                    del logits, logp, loss
                gn = float(sum((p.grad.float() ** 2).sum() for p in params if p.grad is not None).sqrt().cpu())
                res.write(event="mode", mode=mode, first_s=times[0], later_s=times[1:], first_loss=losses[0],
                          last_loss=losses[-1], grad_norm=round(gn, 5), peak_bytes=peak())
            except BaseException as e:  # noqa: BLE001
                res.write(event="mode", mode=mode, error=f"{type(e).__name__}: {e}"[:700])
                if isinstance(e, KeyboardInterrupt):
                    raise
    except BaseException as e:  # noqa: BLE001
        res.fail("train_step", e)
        raise
    finally:
        res.done()


if __name__ == "__main__":
    main()
