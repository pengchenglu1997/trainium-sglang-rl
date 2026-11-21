#!/usr/bin/env python3
"""Does torch.distributed DDP train a LoRA on Neuron, and how does the step time change with
the number of cores?

    torchrun --standalone --nproc-per-node=N 40_ddp_probe.py --device neuron --model DIR \
        --data gsm8k_train.jsonl --results DIR --tag ddpN [--steps 12] [--per-rank-bs 2]
    python 40_ddp_probe.py --device neuron ... --no-ddp --tag single      (one process, no DDP)

Each rank trains on its own slice of a fixed list of GSM8K questions with their reference
solutions (plain next-token loss on the solution), sequences padded to one length. Rank 0
writes one line per step: seconds, loss averaged over the ranks, and at the end a hash of the
LoRA weights. Same seed and same global batch give the same data whatever N is, so the losses
of N = 1, 2, 4, 8 can be laid side by side (they are equal only up to the device's numerics).
The process group follows the Neuron guide "How to use torch.distributed on Neuron": backend
"neuron", initialised before any device tensor; the backend gives each local rank its core
from NEURON_RT_VISIBLE_CORES."""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rl_common as C  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=["neuron", "cuda", "cpu"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--per-rank-bs", type=int, default=2)
    ap.add_argument("--global-bs", type=int, default=0,
                    help="questions per step over all ranks; default per-rank-bs x ranks")
    ap.add_argument("--max-len", type=int, default=384)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--no-ddp", action="store_true")
    a = ap.parse_args()

    if a.device == "neuron":
        os.environ.setdefault("NKI_ENABLE_TRACE_CACHE", "0")
    import torch
    import torch.distributed as dist
    if a.device == "neuron":
        import torch_neuronx  # noqa: F401  (registers the device and the "neuron" backend)
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    use_ddp = not a.no_ddp and "RANK" in os.environ
    res = C.Results(a.results, f"40_ddp_{a.tag}") if rank == 0 else None
    try:
        if use_ddp:
            backend = {"neuron": "neuron", "cuda": "nccl", "cpu": "gloo"}[a.device]
            dist.init_process_group(backend=backend, init_method="env://", rank=rank,
                                    world_size=world)
        dev = {"neuron": "neuron", "cpu": "cpu"}.get(a.device) or f"cuda:{rank % max(1, torch.cuda.device_count())}"

        def sync():
            try:
                if a.device == "neuron":
                    torch_neuronx.synchronize()
                elif a.device == "cuda":
                    torch.cuda.synchronize()
            except Exception:  # noqa: BLE001
                pass

        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
        set_seed(a.seed)                      # same LoRA init on every rank
        tok = AutoTokenizer.from_pretrained(a.model)
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        dtype = torch.float32 if a.device == "cpu" else torch.bfloat16
        model = AutoModelForCausalLM.from_pretrained(a.model, dtype=dtype, low_cpu_mem_usage=True)
        model.to(dev)
        model = get_peft_model(model, LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj",
                            "down_proj"]))
        core = model
        if use_ddp:
            from torch.nn.parallel import DistributedDataParallel
            model = DistributedDataParallel(model)
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.SGD(params, lr=a.lr)

        gbs = a.global_bs or a.per_rank_bs * world
        if gbs % world:
            raise ValueError("the global batch must divide evenly over the ranks")
        prb = gbs // world
        rows = C.read_jsonl(a.data)[: gbs * a.steps]
        if len(rows) < gbs * a.steps:
            raise ValueError("not enough rows in the data file")

        def batch(step):
            mine = rows[step * gbs:(step + 1) * gbs][rank * prb:(rank + 1) * prb]
            ids = torch.full((prb, a.max_len), tok.pad_token_id, dtype=torch.long)
            att = torch.zeros((prb, a.max_len), dtype=torch.long)
            lab = torch.full((prb, a.max_len), -100, dtype=torch.long)
            for i, r in enumerate(mine):
                p = tok(tok.apply_chat_template(C.messages_for(r["question"]), tokenize=False,
                                                add_generation_prompt=True, enable_thinking=False),
                        add_special_tokens=False)["input_ids"][: a.max_len // 2]
                s = tok(r["answer"], add_special_tokens=False)["input_ids"]
                s = (s + [tok.eos_token_id])[: a.max_len - len(p)]
                seq = p + s
                ids[i, :len(seq)] = torch.tensor(seq)
                att[i, :len(seq)] = 1
                lab[i, len(p):len(seq)] = torch.tensor(s)
            return ids.to(dev), att.to(dev), lab.to(dev)

        if res:
            res.write(event="start", tag=a.tag, device=a.device, ddp=use_ddp, world_size=world,
                      per_rank_bs=prb, global_bs=gbs, steps=a.steps, max_len=a.max_len,
                      trainable_params=sum(p.numel() for p in params),
                      visible_cores=os.environ.get("NEURON_RT_VISIBLE_CORES"))
        secs = []
        for step in range(a.steps):
            t0 = time.time()
            ids, att, lab = batch(step)
            opt.zero_grad(set_to_none=True)
            logits = model(input_ids=ids, attention_mask=att).logits
            # mean over each row's solution tokens, then over the rank's rows: with equal rows
            # per rank the average over ranks is the same quantity for every number of ranks
            tok_loss = torch.nn.functional.cross_entropy(
                logits[:, :-1, :].float().reshape(-1, logits.shape[-1]),
                lab[:, 1:].reshape(-1), ignore_index=-100, reduction="none").view(prb, -1)
            m = (lab[:, 1:] != -100).float()
            loss = ((tok_loss * m).sum(1) / m.sum(1).clamp(min=1.0)).mean()
            loss.backward()
            opt.step()
            g = loss.detach().float().clone()
            if use_ddp:
                dist.all_reduce(g)
                g = g / world
            sync()
            g_host = float(g.cpu())          # a host read: the device work of the step is done
            sec = time.time() - t0
            secs.append(sec)
            if res:
                res.write(event="step", step=step + 1, seconds=round(sec, 3),
                          loss=round(g_host, 5))
        h = hashlib.sha1()
        for n, p in core.named_parameters():
            if p.requires_grad:
                h.update(n.encode())
                h.update(p.detach().float().cpu().numpy().tobytes())
        if res:
            res.write(event="summary", tag=a.tag, world_size=world, ddp=use_ddp,
                      step_seconds_after_2=C.describe(secs[2:]),
                      questions_per_second=round(gbs / (sum(secs[2:]) / max(1, len(secs[2:]))), 3)
                      if len(secs) > 2 else None,
                      lora_sha1_rank0=h.hexdigest())
        if use_ddp:
            dist.barrier()
        print(f"PASS rank={rank} world={world}", flush=True)
    except BaseException as e:  # noqa: BLE001
        if res:
            res.fail("ddp", e)
        raise
    finally:
        if res:
            res.done()
        if use_ddp:
            try:
                dist.destroy_process_group()
            except Exception:  # noqa: BLE001
                pass


if __name__ == "__main__":
    main()
