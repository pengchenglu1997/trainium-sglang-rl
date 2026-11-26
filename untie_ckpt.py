#!/usr/bin/env python3
"""Untie a tied-embedding checkpoint for vllm-neuron: a copy with its own lm_head.weight.

Our own short version of the step in HarryZ66/trainium-inference-parity (payload/untie_checkpoint.py,
commit a62cc54, Apache License 2.0, https://github.com/HarryZ66/trainium-inference-parity). The idea is
theirs: vllm-neuron 0.29.0.1.2.0's Qwen3 loader wants a separate `lm_head.weight`, so Qwen3-0.6B
(tie_word_embeddings true, no lm_head in the file) does not start. The copy stores embed_tokens.weight a
second time as lm_head.weight and sets tie_word_embeddings to false: same weights, same outputs.
Differences from theirs: no Hugging Face download (a folder is the input), the copy is built in a
temporary folder and renamed when complete, a finished copy is reused, a sharded checkpoint is handled.

    untie_ckpt.py <model folder> [<output folder>]

Output folder default: "<model folder>-untied" next to it (keep it under MODELS_ROOT: the vllm-neuron
container sees only that tree). Last line printed on stdout = the folder to serve:
  tie_word_embeddings true and no lm_head.weight in the checkpoint -> the copy
  anything else (untied already, or lm_head.weight present)         -> the original folder, unchanged
A reason goes to stderr. Exit 0 on success, 2 on an input problem.

Run it on the host with the Neuron venv (it has torch and safetensors):
  /opt/aws_neuronx_venv_torch_neuronx_2_14_0/bin/python untie_ckpt.py $MODELS_ROOT/Qwen3-0.6B
Needs torch and safetensors (imported only when a copy is made). Nothing else is read or written."""
from __future__ import annotations

import json
import os
import shutil
import sys

MARK = ".untie_ok"          # written last; a folder without it is a half-made copy and is rebuilt
HEAD, EMB = "lm_head.weight", "model.embed_tokens.weight"


def log(msg):
    print(f"untie_ckpt: {msg}", file=sys.stderr)


def untie(src, out=None):
    src = os.path.abspath(src.rstrip("/\\"))
    out = os.path.abspath(out) if out else src + "-untied"
    cfg_path = os.path.join(src, "config.json")
    if not os.path.isfile(cfg_path):
        raise SystemExit(f"untie_ckpt: no config.json in {src}")
    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)
    if not cfg.get("tie_word_embeddings"):
        log("tie_word_embeddings is not true: serving the original")
        return src
    index_path = os.path.join(src, "model.safetensors.index.json")
    files = sorted(f for f in os.listdir(src) if f.endswith(".safetensors"))
    if not files:
        raise SystemExit(f"untie_ckpt: no .safetensors file in {src}")
    index = None
    if os.path.isfile(index_path):
        with open(index_path, encoding="utf-8") as f:
            index = json.load(f)
        if HEAD in index["weight_map"]:
            log("lm_head.weight is already in the checkpoint: serving the original")
            return src
    elif len(files) != 1:
        raise SystemExit(f"untie_ckpt: {len(files)} .safetensors files and no index in {src}")
    if os.path.isfile(os.path.join(out, MARK)):
        log(f"reusing the finished copy {out}")
        return out
    from safetensors.torch import load_file, save_file      # torch is needed from here on

    tmp = out + ".tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(tmp)
    skip = {"config.json"} | (set(files) if index is None else set())
    for name in os.listdir(src):
        p = os.path.join(src, name)
        if os.path.isfile(p) and name not in skip:
            shutil.copy2(p, tmp)
    if index is None:                                        # one file: write it again with the extra tensor
        t = load_file(os.path.join(src, files[0]))
        if HEAD in t:
            shutil.rmtree(tmp)
            log("lm_head.weight is already in the checkpoint: serving the original")
            return src
        if EMB not in t:
            raise SystemExit(f"untie_ckpt: {EMB} is not in {files[0]}")
        t[HEAD] = t[EMB].clone()
        save_file(t, os.path.join(tmp, files[0]), metadata={"format": "pt"})
    else:                                                    # shards: the new tensor gets a shard of its own
        shard = index["weight_map"].get(EMB)
        if shard is None:
            raise SystemExit(f"untie_ckpt: {EMB} is not in the index")
        t = load_file(os.path.join(src, shard))
        extra = "model-lm_head.safetensors"
        save_file({HEAD: t[EMB].clone()}, os.path.join(tmp, extra), metadata={"format": "pt"})
        index["weight_map"][HEAD] = extra
        with open(os.path.join(tmp, "model.safetensors.index.json"), "w", encoding="utf-8") as f:
            json.dump(index, f, indent=2)
    cfg["tie_word_embeddings"] = False
    with open(os.path.join(tmp, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    with open(os.path.join(tmp, MARK), "w", encoding="utf-8") as f:
        f.write(json.dumps({"source": src}) + "\n")
    os.rename(tmp, out)
    log(f"wrote {out}")
    return out


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3) or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        sys.exit(0 if len(sys.argv) == 2 and sys.argv[1] in ("-h", "--help") else 2)
    print(untie(*sys.argv[1:]))
