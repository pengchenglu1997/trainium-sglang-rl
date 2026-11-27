#!/usr/bin/env python3
"""i37 diagnostic patch folders (copies of sglang_neuron_int_topk, each with one edit; not fixes):
  patch_bf16: released topk_topp path (switch unset) rounds its input logits to bf16 (round to nearest even, done with
              int32 bit operations so that no float convert can be folded away) before anything else.
  patch_dead: kernel path (switch on) still calls the kernel and keeps its output alive (0 * its first partial), but the
              shard maximum and winner come from the released _tempered_scores / _first_max.
Writes patch_<x>.diff next to each folder."""
import shutil
import subprocess
from pathlib import Path

W = Path.home() / "w" / "i37"
SRC = Path.home() / "w" / "sglang_rl" / "patches" / "sglang_neuron_int_topk"


def make(name, fname, old, new):
    dst = W / name
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(SRC, dst, symlinks=True)
    p = dst / fname
    s = p.read_text()
    assert s.count(old) == 1, (name, s.count(old))
    p.write_text(s.replace(old, new))
    d = subprocess.run(["diff", "-u", str(SRC / fname), str(p)], capture_output=True, text=True).stdout
    (W / f"{name}.diff").write_text(d)
    print(d)


make("patch_bf16", "sampling.py",
     "    logits, sampled, inverse, score = _tempered_scores(logits, rank, vocabulary_size, temperatures, row_keys)\n"
     "    # Shard candidates via the NKI rotational top-k",
     "    # i37 diagnostic: round the fp32 logits to bf16 values (nearest even) by int32 bit operations\n"
     "    _b = logits.float().contiguous().view(torch.int32)\n"
     "    _b = (_b + (0x7FFF + ((_b >> 16) & 1))) & -65536\n"
     "    logits = _b.view(torch.float32)\n"
     "    logits, sampled, inverse, score = _tempered_scores(logits, rank, vocabulary_size, temperatures, row_keys)\n"
     "    # Shard candidates via the NKI rotational top-k")

make("patch_dead", "rlkit_sampler_topk.py",
     "    token = local_id + rank64 * width                                # v0.1.0 line 373\n",
     "    # i37 diagnostic: winner from the released code; the kernel output is kept alive but does not decide\n"
     "    _, _, _, _score = S._tempered_scores(logits, rank, vocabulary_size, temperatures, row_keys)\n"
     "    value, local_id = S._first_max(_score)\n"
     "    value = value + 0.0 * parts[:, :1, 0]\n"
     "    token = local_id + rank64 * width                                # v0.1.0 line 373\n")
