"""The training half of the RL loop: a text model with one LoRA, a GRPO-style policy-gradient
update on sampled answers, and the export of merged full weights for a server that takes no LoRA.

Devices: "neuron" (TorchNeuron, one core), "cuda", "cpu". On neuron the caller sets
NEURON_RT_VISIBLE_CORES before this module imports torch_neuronx (set_neuron_env below).

Update rule (one optimizer step per call, on-policy, no KL term):
    loss = - mean over sequences of  advantage_i x mean over answer tokens of log p(token)
Sequences are padded to one fixed length so that a compiled-graph device sees one shape.
"""
from __future__ import annotations

import os
import shutil
import time
from pathlib import Path

import rl_common as C

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def set_neuron_env(core, compile_train=False):
    """Before torch_neuronx is imported (the settings of the team's 4B run). With compile_train the
    TORCH_COMPILE=0 switch of that run is left out: the training forward goes through torch.compile."""
    if core is not None:
        os.environ["NEURON_RT_VISIBLE_CORES"] = str(core)
    os.environ["NKI_ENABLE_TRACE_CACHE"] = "0"
    if not compile_train:
        os.environ["TORCH_COMPILE"] = "0"


def _make_window_module(torch, peft_model):
    """The answer-window forward as one module (built lazily so that importing this file does not
    import torch). forward(input_ids, attention_mask, pos_index) -> logits [B, W, V] at the
    positions pos_index [B, W, 1] only: the decoder body runs on the whole padded sequence, the LM
    head only on the selected hidden states."""
    class WindowLM(torch.nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base        # the causal LM with the LoRA layers injected (shared weights)

        def forward(self, input_ids, attention_mask, pos_index):
            out = self.base.model(input_ids=input_ids, attention_mask=attention_mask,
                                  use_cache=False)
            hid = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
            sel = torch.gather(hid, 1, pos_index.expand(-1, -1, hid.shape[-1]))   # [B, W, H]
            return self.base.lm_head(sel)

    return WindowLM(peft_model.get_base_model())


class Trainer:
    def __init__(self, base_dir, device="cuda", lr=1e-5, lora_r=16, lora_alpha=32,
                 max_prompt=320, max_completion=256, micro_bs=4, seed=1, compile_train=False,
                 len_buckets=None, answer_window=False):
        import torch
        self.torch = torch
        self.device_name = device
        if device == "neuron":
            import torch_neuronx  # noqa: F401
            torch._dynamo.config.automatic_dynamic_shapes = False
            self.tn = torch_neuronx
            self.device = "neuron"
        elif device == "cuda":
            self.tn = None
            self.device = "cuda:0"
        else:
            self.tn = None
            self.device = "cpu"
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
        set_seed(seed)
        self.base_dir = Path(base_dir)
        self.tok = AutoTokenizer.from_pretrained(base_dir)
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        dtype = torch.float32 if device == "cpu" else torch.bfloat16
        model = AutoModelForCausalLM.from_pretrained(base_dir, dtype=dtype, low_cpu_mem_usage=True)
        model.to(self.device)
        self.sync()
        cfg = LoraConfig(r=lora_r, lora_alpha=lora_alpha, lora_dropout=0.0,
                         target_modules=LORA_TARGETS, bias="none", task_type="CAUSAL_LM")
        self.model = get_peft_model(model, cfg)
        self.params = [p for p in self.model.parameters() if p.requires_grad]
        # the training forward: under torch.compile on Neuron when asked (52_train_step_probe.py,
        # 2026-11-20: 0.39 s against 0.87 s per micro-batch of 4 x 576, same loss; first call 158 s)
        self.compile_train = bool(compile_train and device == "neuron")
        self.fwd = torch.compile(self.model, backend="neuron") if self.compile_train else self.model
        self.opt = torch.optim.AdamW(self.params, lr=lr, weight_decay=0.0)
        self.max_prompt, self.max_completion = int(max_prompt), int(max_completion)
        self.max_len = self.max_prompt + self.max_completion
        self.micro_bs = int(micro_bs)
        self.updates = 0
        # optional: length buckets (fewer padded positions) and answer-window logits (LM head only
        # on the positions that predict answer tokens); both off by default
        if len_buckets:
            lb = [int(x) for x in len_buckets]
            if lb != sorted(set(lb)) or lb[0] <= 0:
                raise ValueError(f"len_buckets must be strictly ascending positive lengths: {lb}")
            if lb[-1] != self.max_len:
                raise ValueError(f"the last len_bucket {lb[-1]} must equal max_len {self.max_len}")
            self.len_buckets = lb
        else:
            self.len_buckets = None
        self.last_bucket_counts = {}
        self.answer_window = bool(answer_window)
        self.window = self.max_completion + 1
        self.fwd_window = None
        if self.answer_window:
            wm = _make_window_module(torch, self.model)
            self.fwd_window = torch.compile(wm, backend="neuron") if self.compile_train else wm

    # -- device helpers ---------------------------------------------------------------
    def sync(self):
        try:
            if self.device_name == "cuda":
                self.torch.cuda.synchronize()
            elif self.device_name == "neuron":
                self.tn.synchronize()
        except Exception:  # noqa: BLE001
            pass

    def free(self):
        self.sync()
        try:
            if self.device_name == "cuda":
                self.torch.cuda.empty_cache()
            elif self.device_name == "neuron":
                self.tn.empty_cache()
        except Exception:  # noqa: BLE001
            pass

    def sync_check(self) -> dict:
        """Whether the device synchronise call exists and works (written once to the results)."""
        try:
            if self.device_name == "cuda":
                self.torch.cuda.synchronize()
            elif self.device_name == "neuron":
                self.tn.synchronize()
            return {"sync_ok": True, "sync_error": None}
        except Exception as e:  # noqa: BLE001
            return {"sync_ok": False, "sync_error": f"{type(e).__name__}: {e}"[:200]}

    def peak_bytes(self):
        try:
            if self.device_name == "cuda":
                return int(self.torch.cuda.max_memory_allocated())
            if self.device_name == "neuron":
                f = getattr(self.tn, "max_memory_allocated", None)
                return int(f()) if callable(f) else None
        except Exception:  # noqa: BLE001
            return None
        return None

    # -- text ---------------------------------------------------------------------------
    def prompt_text(self, question):
        return self.tok.apply_chat_template(C.messages_for(question), tokenize=False,
                                            add_generation_prompt=True, enable_thinking=False)

    def prompt_tokens(self, question) -> int:
        return len(self.tok(self.prompt_text(question), add_special_tokens=False)["input_ids"])

    def encode(self, question, completion):
        """(ids, n_prompt, n_answer) of prompt + answer + end token, or None when the prompt is
        longer than max_prompt. The answer is cut to max_completion tokens."""
        p = self.tok(self.prompt_text(question), add_special_tokens=False)["input_ids"]
        if len(p) > self.max_prompt:
            return None
        a = self.tok(completion, add_special_tokens=False)["input_ids"]
        if len(a) >= self.max_completion:
            # the sample ran into the token limit: it never chose to stop, so no end token
            a = a[: self.max_completion]
        else:
            a = a + [self.tok.eos_token_id]
        return p + a, len(p), len(a)

    # -- update -------------------------------------------------------------------------
    def update(self, items, temperature=1.0):
        """items: list of (question, completion, advantage). One optimizer step.
        Rows with zero advantage carry no gradient and are left out; when nothing is left the
        step is skipped (reported). temperature: the sampling temperature of the rollout, so
        that the log-probabilities are those of the distribution the answers were drawn from."""
        torch = self.torch
        temperature = float(temperature) if temperature and temperature > 0 else 1.0
        try:
            if self.device_name == "cuda":
                torch.cuda.reset_peak_memory_stats()
            elif self.device_name == "neuron":
                f = getattr(self.tn, "reset_peak_memory_stats", None)
                if callable(f):
                    f()
        except Exception:  # noqa: BLE001
            pass
        rows, too_long = self.encode_items(items)
        out = {"sequences": len(rows), "skipped_prompt_too_long": too_long, "loss": None,
               "mean_token_logp": None, "seconds": 0.0, "micro_batches": 0}
        if not rows:
            out["skipped"] = True
            return out
        t0 = time.time()
        self.opt.zero_grad(set_to_none=True)
        n_seq = len(rows)
        loss_sum, lp_sum, out["micro_batches"] = self.accumulate(rows, temperature, n_seq)
        torch.nn.utils.clip_grad_norm_(self.params, 1.0)
        self.opt.step()
        self.sync()
        self.updates += 1
        out.update(loss=round(loss_sum, 6), mean_token_logp=round(lp_sum / n_seq, 4),
                   seconds=round(time.time() - t0, 3), skipped=False)
        return out

    def encode_items(self, items):
        """items (question, completion, advantage) -> rows ((ids, n_prompt, n_answer), advantage) of
        the ones that carry a gradient, and the count left out for a prompt that is too long."""
        rows, too_long = [], 0
        for q, c, adv in items:
            if c is None or abs(adv) < 1e-12:
                continue
            e = self.encode(q, c)
            if e is None:
                too_long += 1
                continue
            rows.append((e, float(adv)))
        return rows, too_long

    def accumulate(self, rows, temperature, n_seq):
        """Forward and backward of rows in fixed-shape micro-batches; gradients add up in the LoRA
        parameters (the caller zeroes them before and steps after). n_seq: the sequences of the
        whole update, so that shares computed by several workers add up to the same loss.
        Returns (sum of the micro-batch losses, sum of the sequence log-probabilities, micro-batches)."""
        torch = self.torch
        self.model.train()
        pad = self.tok.pad_token_id
        loss_sum = lp_sum = 0.0
        micro = 0
        W = self.window
        # (padded length, rows of the micro-batch); without buckets: the old order, one length
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
            ids = torch.full((self.micro_bs, L), pad, dtype=torch.long)
            att = torch.zeros((self.micro_bs, L), dtype=torch.long)
            tgt = torch.zeros((self.micro_bs, L), dtype=torch.float32)   # 1 on answer tokens
            adv = torch.zeros((self.micro_bs,), dtype=torch.float32)
            if self.answer_window:
                pos = torch.zeros((self.micro_bs, W), dtype=torch.long)     # predicting positions
                tid = torch.zeros((self.micro_bs, W), dtype=torch.long)     # target token ids
                wm = torch.zeros((self.micro_bs, W), dtype=torch.float32)   # weight of each entry
            for i, ((seq, n_p, n_a), a) in enumerate(chunk):
                ids[i, :len(seq)] = torch.tensor(seq)
                att[i, :len(seq)] = 1
                tgt[i, n_p:n_p + n_a] = 1.0
                adv[i] = a
                if self.answer_window:
                    pos[i, :n_a] = torch.arange(n_p - 1, n_p - 1 + n_a)
                    tid[i, :n_a] = torch.tensor(seq[n_p:n_p + n_a])
                    wm[i, :n_a] = 1.0
            if len(chunk) < self.micro_bs:      # filler rows keep the shape; they carry no weight
                att[len(chunk):, 0] = 1
            ids, att, tgt, adv = (x.to(self.device) for x in (ids, att, tgt, adv))
            if self.answer_window:
                pos, tid, wm = (x.to(self.device) for x in (pos, tid, wm))
                logits = self.fwd_window(ids, att, pos.unsqueeze(-1))               # [B, W, V]
                logp = torch.log_softmax(logits.float() / temperature, dim=-1)
                tok_lp = logp.gather(-1, tid.unsqueeze(-1)).squeeze(-1)             # [B, W]
                m = wm
            else:
                logits = self.fwd(input_ids=ids, attention_mask=att).logits
                logp = torch.log_softmax(logits[:, :-1, :].float() / temperature, dim=-1)
                tok_lp = logp.gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)      # [B, L-1]
                m = tgt[:, 1:]
            seq_lp = (tok_lp * m).sum(1) / m.sum(1).clamp(min=1.0)             # mean over answer tokens
            loss = -(adv * seq_lp).sum() / n_seq
            loss.backward()
            loss_sum += float(loss.detach().float().cpu())
            lp_sum += float((seq_lp.detach() * (m.sum(1) > 0)).sum().float().cpu())
            micro += 1
            del logits, logp, tok_lp, loss
        self.last_bucket_counts = counts
        return loss_sum, lp_sum, micro

    # -- generation inside the trainer (the "hf" rollout backend) -----------------------
    def generate(self, question, k, temperature, max_new_tokens):
        torch = self.torch
        self.model.eval()
        self.tok.padding_side = "left"
        text = self.prompt_text(question)
        self.tok.truncation_side = "left"     # never cut the assistant header at the end
        enc = self.tok([text] * k, return_tensors="pt", add_special_tokens=False,
                       padding="max_length", max_length=self.max_prompt, truncation=True)
        enc = {a: b.to(self.device) for a, b in enc.items()}
        kw = dict(max_new_tokens=max_new_tokens, pad_token_id=self.tok.pad_token_id)
        if temperature and temperature > 0:
            kw.update(do_sample=True, temperature=float(temperature), top_p=1.0, top_k=0)
        else:
            kw.update(do_sample=False)
        if self.device_name == "neuron":
            kw.update(cache_implementation="static", disable_compile=True)
        with torch.no_grad():
            g = self.model.generate(**enc, **kw)
        self.sync()
        new = g[:, enc["input_ids"].shape[1]:]
        n_tok = int((new != self.tok.pad_token_id).sum().cpu())
        return self.tok.batch_decode(new, skip_special_tokens=True), n_tok

    # -- weights for the server -----------------------------------------------------------
    def lora_deltas(self):
        """{base weight key: (B @ A) x scaling as float32 on the CPU} for the active adapter."""
        out = {}
        for name, mod in self.model.named_modules():
            la = getattr(mod, "lora_A", None)
            if la is None or "default" not in la:
                continue
            a = mod.lora_A["default"].weight.detach().float().cpu()
            b = mod.lora_B["default"].weight.detach().float().cpu()
            key = name
            for pre in ("base_model.model.",):
                if key.startswith(pre):
                    key = key[len(pre):]
            out[key + ".weight"] = (b @ a) * float(mod.scaling["default"])
        return out

    def save_merged(self, out_dir):
        """A full checkpoint (base weights + LoRA, in the base's own dtype) that a server loads as
        an ordinary model directory. Always merged from the untouched base files, so repeated
        exports do not accumulate rounding. Returns seconds and counts."""
        from safetensors.torch import load_file, save_file
        t0 = time.time()
        out = Path(out_dir)
        if out.exists():
            shutil.rmtree(out)
        out.mkdir(parents=True)
        state = {}
        for f in sorted(self.base_dir.glob("*.safetensors")):
            state.update(load_file(str(f)))
        deltas = self.lora_deltas()
        hit = changed = total = 0
        max_delta = 0.0
        for key, d in deltas.items():
            k = key if key in state else None
            if k is None:
                alt = [s for s in state if s.endswith(key) or key.endswith(s)]
                k = alt[0] if len(alt) == 1 else None
            if k is None:
                continue
            w = state[k]
            new = (w.float() + d).to(w.dtype)
            # the server computes in the base's dtype (bf16): a small update can round away
            changed += int((new != w).sum())
            total += w.numel()
            max_delta = max(max_delta, float(d.abs().max()))
            state[k] = new
            hit += 1
        if hit != len(deltas):
            raise RuntimeError(f"merged {hit} of {len(deltas)} LoRA modules: key names do not match")
        save_file(state, str(out / "model.safetensors"), metadata={"format": "pt"})
        for f in self.base_dir.iterdir():
            if f.is_file() and not f.name.endswith(".safetensors") \
                    and f.name != "model.safetensors.index.json":
                shutil.copy2(f, out / f.name)
        return {"seconds": round(time.time() - t0, 2), "lora_modules": hit,
                "tensors": len(state),
                "share_of_merged_weights_changed": round(changed / total, 6) if total else None,
                "max_abs_delta": max_delta}

    def save_adapter(self, out_dir):
        self.model.save_pretrained(str(out_dir))
