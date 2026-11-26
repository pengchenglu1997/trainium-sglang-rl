#!/usr/bin/env python3
"""Print what the SGLang-Neuron release image has for the LoRA merge kernel (helper of 78_kernel_h1.sh).

Runs inside the image, no arguments. Prints one JSON line and, as the last line, NKI_VERSION=<version or unknown>.
Exit 0 when `import nki` (with nki.isa and nki.language, the three modules lora_merge.py imports) works, else 4.
It imports no torch_neuronx and starts no device work.
"""
import importlib
import importlib.metadata as md
import json
import sys


def version_of(module, dists=()):
    """(__version__ of the imported module, or the metadata version of the first dist that has one, or None)."""
    try:
        m = importlib.import_module(module)
    except Exception as e:  # noqa: BLE001
        return None, f"{type(e).__name__}: {str(e)[:200]}"
    v = getattr(m, "__version__", None)
    if v is None:
        for d in dists:
            try:
                v = md.version(d)
                break
            except Exception:  # noqa: BLE001
                continue
    return (str(v) if v is not None else "unknown"), None


def main():
    info = {"python": sys.version.split()[0]}
    errors = {}
    for module, dists in (("nki", ("nki", "neuronx-nki")), ("neuronxcc", ("neuronx-cc",)), ("numpy", ("numpy",)),
                          ("ml_dtypes", ("ml_dtypes", "ml-dtypes")), ("torch", ("torch",))):
        v, err = version_of(module, dists)
        info[module] = v
        if err:
            errors[module] = err
    for sub in ("nki.isa", "nki.language"):
        try:
            importlib.import_module(sub)
        except Exception as e:  # noqa: BLE001
            errors[sub] = f"{type(e).__name__}: {str(e)[:200]}"
    try:  # distribution version only: importing torch_neuronx would touch the runtime
        info["torch_neuronx_dist"] = md.version("torch-neuronx")
    except Exception:  # noqa: BLE001
        info["torch_neuronx_dist"] = None
    try:
        importlib.import_module("sglang_neuron.backend.stack")
        info["sglang_neuron_backend_stack"] = "imports"
    except Exception as e:  # noqa: BLE001
        info["sglang_neuron_backend_stack"] = None
        errors["sglang_neuron.backend.stack"] = f"{type(e).__name__}: {str(e)[:200]}"
    info["errors"] = errors
    print(json.dumps(info))
    nki_ok = info.get("nki") is not None and "nki.isa" not in errors and "nki.language" not in errors
    print(f"NKI_VERSION={info.get('nki') or 'unknown'}")
    return 0 if nki_ok else 4


if __name__ == "__main__":
    sys.exit(main())
