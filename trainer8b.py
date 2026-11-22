"""Trainer variant for a model that does not fit a core with the 0.6B recipe (Qwen3-8B: 16.4 GB of
weights on a 25.8 GB core): one whole model copy on one core, LoRA on the seven projections, and
four changes against trainer.Trainer, which is imported and subclassed, not copied:

  1. gradient checkpointing per decoder layer (torch.utils.checkpoint, use_reentrant=False), switched
     by checkpoint=True; each layer can also be wrapped in torch.compile (compile_layers=True), which
     does nothing on CPU/CUDA and uses backend "neuron" on a Neuron device;
  2. micro-batch size micro_bs (default 1); the loss of every micro-batch is divided by the
     sequences of the whole update, so the summed gradient is the gradient of trainer.py's loss;
  3. the answer window only goes through the LM head, in chunks of logit_chunk positions, and each
     chunk (LM head, log-softmax, gather) is itself checkpointed when checkpoint=True, so no
     [B, W, vocab] tensor is alive at once (with checkpoint=False autograd keeps every chunk's
     log-softmax output for the backward, so the chunks then save no memory);
  4. the AdamW state lives on the host (host_optimizer=True): float32 master copies of the LoRA
     values on the CPU, gradients are read to the host, the step is taken there, the new values are
     copied back to the device.
A memory report (device free memory through torch.neuron.mem_get_info, or torch.cuda.mem_get_info,
else process RSS) is taken after load, after the forward and after the backward; update() returns
the worst value of each stage in out["mem"].

Same interface as trainer.Trainer (update, accumulate, generate, lora_deltas, save_merged, ...), so
30_rl_loop.py can use it in place of T.Trainer. Not used by ptrainer.py: its workers build
trainer.Trainer directly.
"""
from __future__ import annotations

import os

import trainer as T


class HostAdamW:
    """AdamW whose master values and moment estimates live on the CPU in float32. Same arithmetic as
    torch.optim.AdamW on the parameters themselves when those are float32 on the CPU."""

    def __init__(self, torch, params, lr, weight_decay=0.0):
        self.torch = torch
        self.params = list(params)
        self.master = [torch.nn.Parameter(p.detach().float().cpu().clone()) for p in self.params]
        self.opt = torch.optim.AdamW(self.master, lr=lr, weight_decay=weight_decay)

    def refresh(self):
        """Take the master values from the device values again (after they were set from outside).
        For bfloat16 parameters this is the only way the master follows; float32 ones follow in step()."""
        with self.torch.no_grad():
            for m, p in zip(self.master, self.params):
                m.copy_(p.detach().float().cpu())

    def zero_grad(self, set_to_none=True):
        for p in self.params:
            p.grad = None

    def step(self):
        torch = self.torch
        for m, p in zip(self.master, self.params):
            if p.dtype == torch.float32:      # no precision to keep: follow values set from outside
                with torch.no_grad():
                    m.copy_(p.detach().cpu())
            m.grad = None if p.grad is None else p.grad.detach().float().cpu()
        self.opt.step()
        with torch.no_grad():
            for m, p in zip(self.master, self.params):
                p.copy_(m.detach().to(device=p.device, dtype=p.dtype))


class Trainer8b(T.Trainer):
    def __init__(self, base_dir, device="cuda", micro_bs=1, checkpoint=True, compile_layers=False,
                 logit_chunk=64, host_optimizer=True, mem_hook=None, lr=1e-5, lora_r=16,
                 lora_alpha=32, max_prompt=320, max_completion=256, seed=1, compile_train=False,
                 len_buckets=None, answer_window=True):
        # compile_train (the loop's --train-compile) means compile_layers here: the whole-model
        # compile of the base class is never used, so super() gets compile_train=False.
        # answer_window is accepted for interface compatibility; this trainer always uses the window.
        super().__init__(base_dir, device=device, lr=lr, lora_r=lora_r, lora_alpha=lora_alpha,
                         max_prompt=max_prompt, max_completion=max_completion, micro_bs=micro_bs,
                         seed=seed, compile_train=False, len_buckets=len_buckets,
                         answer_window=False)
        torch = self.torch
        self.checkpoint = bool(checkpoint)
        self.compile_layers = bool(compile_layers or compile_train)
        self.logit_chunk = max(1, int(logit_chunk))
        self.mem_hook = mem_hook
        self.mem = {}
        self.base = self.model.get_base_model()          # the causal LM with the LoRA layers injected
        self._wrap_layers()
        if host_optimizer:
            lr_ = self.opt.param_groups[0]["lr"]
            self.opt = HostAdamW(torch, self.params, lr_, 0.0)
        self.host_optimizer = bool(host_optimizer)
        self.mem_note("after_load")

    # -- per-layer checkpoint and compile ----------------------------------------------------
    def _wrap_layers(self):
        torch = self.torch
        from torch.utils.checkpoint import checkpoint as ckpt
        use_compile = self.compile_layers and self.device_name == "neuron"   # no-op on CPU / CUDA
        for layer in self.base.model.layers:
            if getattr(layer, "_t8b_wrapped", False):
                continue
            fn = layer.forward
            if use_compile:
                fn = torch.compile(fn, backend="neuron")
            if self.checkpoint:
                def fwd(*args, _fn=fn, _layer=layer, **kw):
                    if torch.is_grad_enabled() and _layer.training:
                        return ckpt(_fn, *args, use_reentrant=False, **kw)
                    return _fn(*args, **kw)
            else:
                fwd = fn
            layer.forward = fwd
            layer._t8b_wrapped = True

    # -- memory report -------------------------------------------------------------------------
    def mem_report(self, stage):
        torch = self.torch
        d = {"stage": stage}
        self.sync()
        try:
            f = None
            if self.device_name == "neuron":
                f = getattr(getattr(torch, "neuron", None), "mem_get_info", None)
            elif self.device_name == "cuda":
                f = torch.cuda.mem_get_info
            if callable(f):
                free, total = f()
                d.update(device_free_bytes=int(free), device_total_bytes=int(total))
        except Exception:  # noqa: BLE001
            pass
        if "device_free_bytes" not in d:
            try:
                import psutil
                d["rss_bytes"] = int(psutil.Process(os.getpid()).memory_info().rss)
            except Exception:  # noqa: BLE001
                try:
                    with open("/proc/self/status") as fh:
                        for line in fh:
                            if line.startswith("VmRSS:"):
                                d["rss_bytes"] = int(line.split()[1]) * 1024
                except Exception:  # noqa: BLE001
                    d["rss_bytes"] = None
        return d

    def mem_note(self, stage):
        """Take a report and keep the worst one of the stage (lowest free memory / highest RSS)."""
        d = self.mem_report(stage)
        old = self.mem.get(stage)
        if old is None:
            self.mem[stage] = d
        elif "device_free_bytes" in d:
            if d["device_free_bytes"] < old.get("device_free_bytes", d["device_free_bytes"] + 1):
                self.mem[stage] = d
        elif (d.get("rss_bytes") or 0) > (old.get("rss_bytes") or 0):
            self.mem[stage] = d
        if self.mem_hook is not None:
            self.mem_hook(d)
        return d

    def update(self, items, temperature=1.0):
        load = self.mem.get("after_load")
        self.mem = {"after_load": load} if load else {}
        out = super().update(items, temperature)
        out["mem"] = dict(self.mem)
        return out

    # -- forward pieces ------------------------------------------------------------------------
    def window_logp(self, ids, att, pos, tid, temperature):
        """Log-probabilities [B, W] of the target ids tid at the positions pos ([B, W]): decoder body on
        the whole padded sequence (layers checkpointed), then LM head + log-softmax + gather chunk by
        chunk."""
        torch = self.torch
        from torch.utils.checkpoint import checkpoint as ckpt
        out = self.base.model(input_ids=ids, attention_mask=att, use_cache=False)
        hid = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        sel = torch.gather(hid, 1, pos.unsqueeze(-1).expand(-1, -1, hid.shape[-1]))   # [B, W, H]
        head = self.base.lm_head

        def chunk_lp(h, t):
            logp = torch.log_softmax(head(h).float() / temperature, dim=-1)
            return logp.gather(-1, t.unsqueeze(-1)).squeeze(-1)

        parts = []
        W = sel.shape[1]
        for s in range(0, W, self.logit_chunk):
            h, t = sel[:, s:s + self.logit_chunk], tid[:, s:s + self.logit_chunk]
            if self.checkpoint and torch.is_grad_enabled():
                parts.append(ckpt(chunk_lp, h, t, use_reentrant=False))
            else:
                parts.append(chunk_lp(h, t))
        return torch.cat(parts, dim=1)

    def accumulate(self, rows, temperature, n_seq):
        """As trainer.Trainer.accumulate (same arguments, same return, same loss), micro-batch by
        micro-batch, with the window forward above."""
        torch = self.torch
        self.model.train()
        pad = self.tok.pad_token_id
        loss_sum = lp_sum = 0.0
        micro = 0
        W = self.window
        if self.len_buckets:
            groups = {b: [] for b in self.len_buckets}
            for r in rows:
                n = len(r[0][0])
                b = next((b for b in self.len_buckets if n <= b), None)
                if b is None:
                    raise ValueError(f"a sequence of {n} tokens exceeds the last bucket {self.len_buckets[-1]}")
                groups[b].append(r)
            batches = [(b, g[s:s + self.micro_bs]) for b, g in sorted(groups.items())
                       for s in range(0, len(g), self.micro_bs)]
        else:
            batches = [(self.max_len, rows[s:s + self.micro_bs])
                       for s in range(0, len(rows), self.micro_bs)]
        counts = {}
        for L, chunk in batches:
            counts[L] = counts.get(L, 0) + 1
            B = self.micro_bs
            ids = torch.full((B, L), pad, dtype=torch.long)
            att = torch.zeros((B, L), dtype=torch.long)
            adv = torch.zeros((B,), dtype=torch.float32)
            pos = torch.zeros((B, W), dtype=torch.long)
            tid = torch.zeros((B, W), dtype=torch.long)
            wm = torch.zeros((B, W), dtype=torch.float32)
            for i, ((seq, n_p, n_a), a) in enumerate(chunk):
                ids[i, :len(seq)] = torch.tensor(seq)
                att[i, :len(seq)] = 1
                adv[i] = a
                pos[i, :n_a] = torch.arange(n_p - 1, n_p - 1 + n_a)
                tid[i, :n_a] = torch.tensor(seq[n_p:n_p + n_a])
                wm[i, :n_a] = 1.0
            if len(chunk) < B:      # filler rows keep the shape; they carry no weight
                att[len(chunk):, 0] = 1
            ids, att, adv, pos, tid, wm = (x.to(self.device) for x in (ids, att, adv, pos, tid, wm))
            tok_lp = self.window_logp(ids, att, pos, tid, temperature)           # [B, W]
            seq_lp = (tok_lp * wm).sum(1) / wm.sum(1).clamp(min=1.0)
            loss = -(adv * seq_lp).sum() / n_seq
            self.mem_note("after_forward")
            loss.backward()
            self.mem_note("after_backward")
            loss_sum += float(loss.detach().float().cpu())
            lp_sum += float((seq_lp.detach() * (wm.sum(1) > 0)).sum().float().cpu())
            micro += 1
            del tok_lp, loss
        self.last_bucket_counts = counts
        return loss_sum, lp_sum, micro
