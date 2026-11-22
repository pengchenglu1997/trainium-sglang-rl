"""CPU check of trainer8b.Trainer8b against trainer.Trainer (W0.3 of the Qwen3 experiment plan).

Builds a tiny random Qwen3-shaped causal LM (3 layers, hidden 64, no download) and a word-level
tokenizer in a temporary folder, loads it into both trainers through their normal constructors (so
the LoRA is attached exactly as trainer.py does it), gives both the same non-zero LoRA values, and
checks in float32 on the CPU, on the same rows:
  - the loss of accumulate() is the same;
  - the gradient of every LoRA parameter from Trainer8b (checkpointing on, chunked logits, micro-batch
    1 and 2, host optimizer) equals trainer.py's within 1e-5 relative (norm of the difference over the
    norm of the reference gradient), and no reference gradient is zero;
  - one update() (clip + one optimizer step) gives the same LoRA values (same relative bound);
  - the memory report has its three stages.
Also reports (information, not a pass condition) the peak size of one [B, W, vocab] logits tensor
against the chunk size. Prints PASS / FAILED lines and a last line with the counts.

    python 65_trainer8b_cpu_test.py [--results DIR]
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

TOL = 1e-5
COUNTS = {"pass": 0, "fail": 0}
RES = None


def check(name, ok, detail=""):
    COUNTS["pass" if ok else "fail"] += 1
    print(f"{'PASS' if ok else 'FAILED'}: {name} {detail}".rstrip())
    if RES is not None:
        RES.write(event="check", name=name, ok=bool(ok), detail=str(detail))


def build_dir(tmp, tied, vocab=128):
    import torch
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast, Qwen3Config
    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=vocab, hidden_size=64, intermediate_size=128, num_hidden_layers=3,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                      max_position_embeddings=1024, tie_word_embeddings=tied)
    model = AutoModelForCausalLM.from_config(cfg)
    model.save_pretrained(tmp)
    words = ["<pad>", "<eos>", "<unk>"] + [f"w{i}" for i in range(vocab - 3)]
    tk = Tokenizer(models.WordLevel({w: i for i, w in enumerate(words)}, unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=tk, pad_token="<pad>", eos_token="<eos>",
                            unk_token="<unk>").save_pretrained(tmp)


def make_rows(n, max_prompt, max_completion, vocab, seed):
    import random
    rnd = random.Random(seed)
    rows = []
    for i in range(n):
        n_p = rnd.randint(max(2, max_prompt // 3), max_prompt)
        n_a = rnd.randint(1, max_completion)
        seq = [rnd.randint(3, vocab - 1) for _ in range(n_p + n_a)]
        adv = rnd.choice([-1, 1]) * rnd.uniform(0.3, 1.5)
        rows.append(((seq, n_p, n_a), adv))
    return rows


def rel(torch, a, b):
    return float((a - b).norm() / b.norm().clamp(min=1e-30))


def case(torch, tmp, label, max_prompt, max_completion, n_rows, micro_bs, chunk, temperature, vocab):
    import trainer as T
    import trainer8b as T8
    kw = dict(device="cpu", lr=1e-3, lora_r=4, lora_alpha=8, max_prompt=max_prompt,
              max_completion=max_completion, seed=1)
    ref = T.Trainer(tmp, micro_bs=micro_bs if micro_bs > 1 else 2, **kw)
    t8 = T8.Trainer8b(tmp, micro_bs=micro_bs, checkpoint=True, logit_chunk=chunk,
                      host_optimizer=True, **kw)
    # same non-zero LoRA values in both (B starts at zero, which would hide the gradient of A)
    g = torch.Generator().manual_seed(7)
    with torch.no_grad():
        for p, q in zip(ref.params, t8.params):
            v = torch.randn(p.shape, generator=g) * 0.05
            p.copy_(v)
            q.copy_(v)
    rows = make_rows(n_rows, max_prompt, max_completion, vocab, seed=3)
    n_seq = len(rows)

    # gradients of the unsplit reference (micro-batch of the whole set when it fits)
    ref.micro_bs = n_rows
    ref.opt.zero_grad(set_to_none=True)
    l_ref, lp_ref, _ = ref.accumulate(rows, temperature, n_seq)
    g_ref = [p.grad.detach().clone() for p in ref.params]
    t8.opt.zero_grad(set_to_none=True)
    l_8, lp_8, mic = t8.accumulate(rows, temperature, n_seq)
    g_8 = [p.grad.detach().clone() for p in t8.params]
    expect_mic = -(-n_rows // micro_bs)
    check(f"{label}: micro-batches", mic == expect_mic, f"({mic} of {expect_mic} expected)")
    check(f"{label}: loss", abs(l_ref - l_8) <= TOL * max(abs(l_ref), 1e-12),
          f"(ref {l_ref:.7f}, 8b {l_8:.7f})")
    check(f"{label}: mean log-prob", abs(lp_ref - lp_8) <= TOL * max(abs(lp_ref), 1e-12),
          f"(ref {lp_ref:.6f}, 8b {lp_8:.6f})")
    zero = sum(1 for x in g_ref if float(x.abs().max()) == 0.0)
    check(f"{label}: no reference gradient is zero", zero == 0, f"({zero} of {len(g_ref)} zero)")
    worst = max(rel(torch, a, b) for a, b in zip(g_8, g_ref))
    n_bad = sum(1 for a, b in zip(g_8, g_ref) if rel(torch, a, b) > TOL)
    check(f"{label}: LoRA gradients ({len(g_ref)} tensors) within {TOL}", n_bad == 0,
          f"(worst relative difference {worst:.2e})")

    # one full update through update() on both (clip + optimizer step), rows injected
    for tr in (ref, t8):
        tr.encode_items = lambda items, _rows=rows: (list(_rows), 0)
    ref.micro_bs = n_rows
    o_ref = ref.update([("q", "a", 1.0)], temperature)
    o_8 = t8.update([("q", "a", 1.0)], temperature)
    worst_v = max(rel(torch, q.detach(), p.detach()) for p, q in zip(ref.params, t8.params))
    check(f"{label}: LoRA values after one update", worst_v <= TOL, f"(worst relative difference {worst_v:.2e})")
    check(f"{label}: update loss", abs(o_ref["loss"] - o_8["loss"]) <= 1e-5 * max(abs(o_ref["loss"]), 1e-9) + 1e-6,
          f"(ref {o_ref['loss']}, 8b {o_8['loss']})")
    m = o_8.get("mem", {})
    check(f"{label}: memory report stages", all(k in m for k in ("after_load", "after_forward", "after_backward")),
          f"({sorted(m)}; rss after backward {m.get('after_backward', {}).get('rss_bytes')})")
    return ref, t8


def main():
    global RES
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=None)
    a = ap.parse_args()
    if a.results:
        import rl_common as C
        RES = C.Results(a.results, "65_trainer8b_cpu_test")
    import torch
    import transformers
    import peft
    torch.set_default_dtype(torch.float32)
    print(f"torch {torch.__version__}, transformers {transformers.__version__}, peft {peft.__version__}")
    vocab = 128
    with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
        build_dir(d1, tied=False, vocab=vocab)
        build_dir(d2, tied=True, vocab=vocab)
        # small shapes: micro-batch 1 and 2 (the last micro-batch of 5 rows is a filler-padded one), two temperatures
        case(torch, d1, "untied, mb1, chunk 4", 12, 8, 5, 1, 4, 1.0, vocab)
        case(torch, d1, "untied, mb2 (filler row), chunk 3", 12, 8, 5, 2, 3, 0.7, vocab)
        case(torch, d2, "tied head, mb1, chunk 5", 12, 8, 4, 1, 5, 1.0, vocab)
        # the real training shape: 320 + 256 = 576 positions, window of 257
        ref, t8 = case(torch, d1, "length 576, mb1, chunk 64", 320, 256, 3, 1, 64, 1.0, vocab)
        check("checkpoint wrappers installed on every layer",
              all(getattr(l, "_t8b_wrapped", False) for l in t8.base.model.layers),
              f"({len(t8.base.model.layers)} layers)")
        # checkpointing off must also match (the flag, not only the default)
        import trainer8b as T8
        t_off = T8.Trainer8b(d1, device="cpu", micro_bs=1, checkpoint=False, logit_chunk=7,
                             host_optimizer=False, lr=1e-3, lora_r=4, lora_alpha=8, max_prompt=12,
                             max_completion=8, seed=1)
        t_on = T8.Trainer8b(d1, device="cpu", micro_bs=1, checkpoint=True, logit_chunk=7,
                            host_optimizer=False, compile_layers=True, lr=1e-3, lora_r=4, lora_alpha=8,
                            max_prompt=12, max_completion=8, seed=1)
        # compile_layers=True on CPU: the flag is kept, but no layer function is a torch.compile wrapper
        inner = [(l.forward.__kwdefaults__ or {}).get("_fn", l.forward) for l in t_on.base.model.layers]
        n_comp = sum(1 for f in inner if hasattr(f, "_torchdynamo_orig_callable"))
        check("compile_layers is a no-op on CPU", t_on.compile_layers is True and n_comp == 0,
              f"(flag {t_on.compile_layers}, compiled layers {n_comp} of {len(inner)})")
        gen =torch.Generator().manual_seed(11)
        with torch.no_grad():
            for p, q in zip(t_off.params, t_on.params):
                v = torch.randn(p.shape, generator=gen) * 0.05
                p.copy_(v)
                q.copy_(v)
        rows = make_rows(4, 12, 8, vocab, seed=5)
        calls = {}
        for name, t in (("off", t_off), ("on", t_on)):
            calls[name] = [0]
            # a pre-hook: the recompute of a checkpoint stops inside the layer once it has what it
            # needs, so a hook after the MLP would not see it
            t.base.model.layers[1].mlp.register_forward_pre_hook(
                lambda m, i, _c=calls[name]: _c.__setitem__(0, _c[0] + 1))
            t.opt.zero_grad(set_to_none=True)
            t.accumulate(rows, 1.0, len(rows))
        check("checkpointing recomputes a layer in the backward (2 forward calls per micro-batch, 1 without)",
              calls["off"][0] == 4 and calls["on"][0] == 8,
              f"(layer 1 MLP calls over 4 micro-batches: off {calls['off'][0]}, on {calls['on'][0]})")
        worst = max(rel(torch, p.grad, q.grad) for p, q in zip(t_on.params, t_off.params))
        check("checkpoint off vs on (compile_layers flag set), same gradients", worst <= TOL,
              f"(worst relative difference {worst:.2e})")
        # logits memory: the largest tensor of the chunked head against the unchunked one (arithmetic, not measured)
        W, V = 257, 151936
        print(f"info: Qwen3 answer window {W} positions x vocab {V} x 4 bytes = {W * V * 4 / 1e6:.1f} MB per row "
              f"unchunked; chunk 64 = {64 * V * 4 / 1e6:.1f} MB per row (arithmetic from these two sizes)")
    line = f"RESULT: {COUNTS['pass']} passed, {COUNTS['fail']} failed"
    if RES is not None:
        RES.write(event="summary", **COUNTS)
        RES.done()
    print(line)
    return 1 if COUNTS["fail"] else 0


if __name__ == "__main__":
    sys.exit(main())
