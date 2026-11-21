# SPDX-License-Identifier: Apache-2.0
"""The class SGLang builds for a model's checkpoint. It sets up the parallel layout, builds this server's role
graphs (both on an aggregated server), follows each with the device sampler unless the graph samples in its own
forward or the host samples, and loads their weights. A model's host names itself and builds its graphs."""

import logging
import shlex
from types import MappingProxyType
from typing import Iterable, Mapping, Tuple

import torch
from torch import nn

from sglang_neuron import envs, sampling, sglang_tree
from sglang_neuron.model_executor.runtime_init import neuron_device
from sglang_neuron.models.base import NeuronModel, segmented_prefill_cache_policy
from sglang_neuron.models.phase import model_phase
from sglang_neuron.parallel.state import initialize_neuron_parallel_state

logger = logging.getLogger(__name__)


class ModelHost(NeuronModel):
    """``name`` names the model in messages; ``build_graph(phase, config)`` builds one role graph on the host."""

    name: str
    # whether the prefill also compiles the segment tiles below the chunk, for the last piece of a prompt
    # (graph_runner.BucketConfig)
    prefill_ladder = True
    # whether the decode also compiles graphs over a quarter and a half of the context, for batches whose rows are
    # short (graph_runner.BucketConfig); a model opts in once its decode attention reads only its graph's context
    decode_context_ladder = False
    # the prefill attention is the segmented kernel, whose KV segment tile is the prefill token bucket
    # (graph_runner.check_prefill_tiles); a model with a prefill attention of its own sets it False
    segmented_prefill = True
    # whether the prefill graph's window holds the request's cached prior beside its new tokens, so a step takes the
    # bucket of both (NeuronModelRuntime.plan_step)
    prefill_holds_prior = False
    # component DP: per component (embedding, LM head, MLP; parallel.groups.COMPONENTS), the number of consecutive DP
    # replicas whose attention-TP ranks it shards over; no model declares one yet
    component_dp: Mapping[str, int] = MappingProxyType({})

    def __init__(self, config, quant_config=None, prefix: str = "", **kwargs):
        super().__init__()
        self.check_launch(config)
        self.hf_config = config
        pc = sglang_tree.parallel_config()
        self.tp_size, self.ep_degree = int(pc.tp_size), int(pc.ep_size or 1)
        self.dp_size = sglang_tree.attention_dp_size(pc)
        # one attention replica per DP rank; experts span the replicas through EP, a component_dp entry that many
        initialize_neuron_parallel_state(ep_degree=self.ep_degree, dp_size=self.dp_size, component_dp=self.component_dp)
        self._set_stage(config)
        self.phase = model_phase()
        # a prefilling role keeps its forwards on the chunk grid the segmented kernel's tiles are compiled for
        segmented_prefill_cache_policy(self.phase)
        phases = ("prefill", "decode") if self.phase == "all" else (self.phase,)
        # Built on the host: load_weights replaces each parameter with its device tensor, so the weights reach the
        # device once. Building on the device would hold two copies at the peak and leave no room for the LNC2
        # scratchpad at the top of HBM, and the NEFF load would fail.
        with torch.device("cpu"):
            self.model_graphs = {phase: self.build_graph(phase, config) for phase in phases}
        # an aggregated server's default graph is its decode
        self.model_graph = self.model_graphs[phases[-1]]
        mode = sampling.sampler_mode()
        # the device sampler reads each rank's vocabulary shard of the logits, the host sampler the gathered logits; a
        # non-last pipeline stage has no LM head and returns its boundary
        samples_on_device = self.is_last_stage and mode != sampling.HOST_SAMPLER
        for graph in self.model_graphs.values():
            if samples_on_device:
                graph.lm_head.gather_output = False
        self.execution_models = {phase: sampled_execution(graph, mode) if samples_on_device else graph
                                 for phase, graph in self.model_graphs.items()}
        self.execution_model = self.execution_models[phases[-1]]
        self.kv_spec = self.model_graph.get_kv_spec()
        self.runtime = None
        self._weights_loaded = False
        logger.info("%s (%s) world=%d dp=%d ep=%d layers=%d%s", self.name, self.phase, self.tp_size, self.dp_size,
                    self.ep_degree, config.num_hidden_layers,
                    "" if self.stage is None else f" stage {self.stage.pp_rank}/{self.stage.pp_size} layers "
                    f"[{self.start_layer}, {self.end_layer})")

    def update_weights_in_place(self, path: str) -> None:
        """EXPERIMENT (2026-11-20), not upstream. New weights for a model that is already serving, without a restart:
        ``path`` is a checkpoint of the same model, or a PEFT LoRA adapter folder for it (adapter_config.json).

        A checkpoint: this rank's shards are rebuilt on the host by the same loaders as the first load (a fresh CPU
        graph), then written with copy_ into the existing device parameters. No parameter object is replaced, so the
        executors' cached input views stay valid; nothing is compiled, nothing is warmed. The caller (the
        update_weights_from_disk hook in general_plugin) drains the device queue first."""
        import os
        import time

        if not self._weights_loaded:
            raise RuntimeError(f"{self.name}: update_weights_in_place before the first load")
        if os.path.exists(os.path.join(path, "adapter_config.json")):
            return self._experiment_apply_adapter(path)
        t0 = time.time()
        (first_phase, first), *_ = self.model_graphs.items()
        with torch.device("cpu"):
            staged = self.build_graph(first_phase, self.hf_config)
        staged.load_weights(path, torch.device("cpu"))
        t1 = time.time()
        new = dict(staged.named_parameters())
        live = dict(first.named_parameters())
        missing = sorted(set(live) - set(new))
        if missing:
            raise ValueError(f"in-place update: {len(missing)} live parameters have no staged tensor, e.g. {missing[:3]}")
        for name, p in live.items():
            q = new[name]
            if q.shape != p.shape or q.dtype != p.dtype:
                raise ValueError(f"in-place update: {name} is {tuple(p.shape)} {p.dtype} live, "
                                 f"{tuple(q.shape)} {q.dtype} staged")
        nbytes = 0
        with torch.no_grad():
            for name, p in live.items():
                p.data.copy_(new[name])
                nbytes += p.numel() * p.element_size()
        try:
            import torch_neuronx
            torch_neuronx.synchronize()
        except Exception:  # noqa: BLE001
            pass
        state = getattr(self, "_experiment_adapter_state", None)
        if state is not None:               # a full update is the new base for later adapters
            state["base"].clear()
            state.get("base_dev", {}).clear()
        t2 = time.time()
        logger.info("[neuron] EXPERIMENT in-place weight update from %s: %d tensors, %d bytes, stage %.2fs, copy %.2fs",
                    path, len(live), nbytes, t1 - t0, t2 - t1)

    def _experiment_apply_adapter(self, path: str) -> None:
        """EXPERIMENT, not upstream. ``path`` holds a PEFT LoRA adapter (adapter_config.json,
        adapter_model.safetensors) for the model this server was started on. Only the adapter is read. For every
        live parameter built from an adapted checkpoint matrix, the delta scale * (B @ A) is pushed through that
        parameter's own load-time loader (so sharding, QKV fusion and transposes are the model's own code), added
        to a host copy of this rank's base shard taken at the first adapter update, and written in place.
        Each update starts from that base copy, so adapters replace each other and do not add up.
        Limits of this sketch: every checkpoint matrix of an adapted parameter must be in the adapter (q, k and v
        together; gate and up together); the base copy is whatever the server holds at the first adapter update;
        the delta is formed in full on the host before it is cut to this rank."""
        import json
        import os
        import time

        from safetensors.torch import load_file

        from sglang_neuron.utils.weight_loader import get_weight_loader

        t0 = time.time()
        with open(os.path.join(path, "adapter_config.json")) as f:
            cfg = json.load(f)
        scale = float(cfg["lora_alpha"]) / float(cfg["r"])
        raw = load_file(os.path.join(path, "adapter_model.safetensors"))
        a_of, b_of = {}, {}
        for key, tensor in raw.items():
            k = key[len("base_model.model."):] if key.startswith("base_model.model.") else key
            for tag, table in ((".lora_A.weight", a_of), (".lora_B.weight", b_of)):
                if k.endswith(tag):
                    table[k[:-len(tag)] + ".weight"] = tensor.float()
        (first_phase, first), *_ = self.model_graphs.items()
        live = dict(first.named_parameters())
        state = getattr(self, "_experiment_adapter_state", None)
        if state is None:
            with torch.device("cpu"):
                staged = self.build_graph(first_phase, self.hf_config)
            mappings = staged.hf_weight_mappings()
            loaders = {name: get_weight_loader(p) for name, p in staged.named_parameters()}
            state = self._experiment_adapter_state = {"mappings": mappings, "loaders": loaders, "base": {}}
        mappings, loaders, base = state["mappings"], state["loaders"], state["base"]
        rank = first.tp_group.rank_in_group
        base_dev = state.setdefault("base_dev", {})

        def sync():
            try:
                import torch_neuronx
                torch_neuronx.synchronize()
            except Exception:  # noqa: BLE001
                pass

        def apply(on_device: bool):
            """One pass over the adapted parameters. on_device: B @ A, the loader's cut and the sum with the base
            all run on the NeuronCore (a second copy of the adapted base shards stays on the device); otherwise on
            the host from the host copy, with one host-to-device write per parameter."""
            touched, nbytes, t_delta, t_copy = 0, 0, 0.0, 0.0
            diag_rows = []
            fa ={k: v.to(live_device) for k, v in a_of.items()} if on_device else a_of
            fb = {k: v.to(live_device) for k, v in b_of.items()} if on_device else b_of
            with torch.no_grad():
                for name, keys in mappings.items():
                    keys = [keys] if isinstance(keys, str) else list(keys)
                    hit = [k in a_of for k in keys]
                    if not any(hit) and name not in base:
                        continue
                    if any(hit) and not all(hit):
                        raise NotImplementedError(f"adapter covers only part of {name}: {keys}")
                    p = live[name]
                    if name not in base:                    # before the first write to this parameter
                        base[name] = p.detach().float().cpu().clone()
                    if on_device and name not in base_dev:
                        base_dev[name] = base[name].to(p.dtype).to(live_device)
                    t = time.time()
                    source = base_dev[name].float() if on_device else base[name]
                    if any(hit):
                        delta = loaders[name].load([scale * (fb[k] @ fa[k]) for k in keys], rank)
                        if delta.shape != p.shape:
                            raise ValueError(f"adapter delta for {name} is {tuple(delta.shape)}, live {tuple(p.shape)}")
                        new = (source + delta).to(p.dtype)
                    else:                   # adapted before, not in this adapter: back to the base
                        new = source.to(p.dtype)
                    t_delta += time.time() - t
                    if on_device and diag_all and any(hit):
                        # the same parameter on the host, to find which ones the device path gets wrong
                        h_new = (base[name] + loaders[name].load([scale * (b_of[k] @ a_of[k]) for k in keys],
                                                                 rank)).to(p.dtype)
                        dd = (new.cpu().float() - h_new.float()).abs()
                        share = float((dd > 0).float().mean())
                        diag_rows.append((share, float(dd.max()), name, tuple(p.shape)))
                    t = time.time()
                    if on_device and write_mode == "host":      # device result, written from the host
                        p.data.copy_(new.cpu())
                    elif on_device and write_mode == "add":     # in-place arithmetic instead of a device copy
                        p.data.zero_()
                        p.data.add_(new)
                    else:
                        p.data.copy_(new)
                    t_copy += time.time() - t
                    touched += 1
                    nbytes += p.numel() * p.element_size()
            sync()
            if diag_rows:
                diag_rows.sort(reverse=True)
                bad = [r for r in diag_rows if r[0] > 1e-3]
                kinds = {}
                for share, mx, nm, shp in bad:
                    kinds[nm.split(".")[-1]] = kinds.get(nm.split(".")[-1], 0) + 1
                logger.info("[neuron] EXPERIMENT adapter diagnostic rank %d: %d of %d parameters differ from the host "
                            "result in more than 0.1%% of entries; by kind %s; worst %s", rank, len(bad), len(diag_rows),
                            kinds, [(nm, shp, round(share, 4), round(mx, 5)) for share, mx, nm, shp in diag_rows[:4]])
            return touched, nbytes, t_delta, t_copy

        live_device = next(iter(live.values())).device
        # Host by default: on 2026-11-20 the host path gave token logprobs identical to a fresh start on the merged
        # checkpoint, the device path did not (off by up to 1.1), cause not found (the same operations compared
        # outside the server agree to 1e-9). RLKIT_ADAPTER_ON_DEVICE=1 turns the device path on and
        # logs, for the first adapted parameter, how its result differs from the host's.
        want_device = os.environ.get("RLKIT_ADAPTER_ON_DEVICE", "0") == "1"
        # RLKIT_DEVICE_WRITE: copy (device tensor into the parameter, the form that did not match), host, add
        write_mode = os.environ.get("RLKIT_DEVICE_WRITE", "copy")
        diag_all = os.environ.get("RLKIT_ADAPTER_DIAG", "0") == "1"     # compare every parameter with the host result
        if want_device and state.get("on_device", True):
            try:
                with torch.no_grad():
                    name0, keys0 = next((n, k) for n, k in mappings.items()
                                        if all(x in a_of for x in ([k] if isinstance(k, str) else k)))
                    keys0 = [keys0] if isinstance(keys0, str) else list(keys0)
                    p0 = live[name0]
                    b0 = base[name0] if name0 in base else p0.detach().float().cpu()
                    d_host = loaders[name0].load([scale * (b_of[k] @ a_of[k]) for k in keys0], rank)
                    d_dev = loaders[name0].load([scale * (b_of[k].to(live_device) @ a_of[k].to(live_device))
                                                 for k in keys0], rank)
                    n_host = (b0 + d_host).to(p0.dtype)
                    n_dev = (b0.to(p0.dtype).to(live_device).float() + d_dev).to(p0.dtype).cpu()
                    dd = (d_dev.cpu().float() - d_host).abs()
                    nd = (n_dev.float() - n_host.float()).abs()
                    logger.info("[neuron] EXPERIMENT adapter diagnostic %s rank %d: delta dtype dev %s, delta max diff %.3e "
                                "(mean |delta| %.3e), final values differing %.4f%%, final max diff %.3e",
                                name0, rank, d_dev.dtype, float(dd.max()), float(d_host.abs().mean()),
                                100.0 * float((nd > 0).float().mean()), float(nd.max()))
            except Exception as e:  # noqa: BLE001
                logger.warning("[neuron] EXPERIMENT adapter diagnostic failed: %s: %s", type(e).__name__, str(e)[:200])
        where = "device" if (want_device and state.get("on_device", True)) else "host"
        if os.environ.get("RLKIT_DUMMY_DEVICE_OP", "0") == "1":
            # Hypothesis test (2026-11-20): the device path writes the same values as the host path (diagnostic: every
            # parameter equal up to single ulps) and still serves different logprobs. Does running eager device
            # operations in this process change what the compiled graphs compute? Host path plus one unrelated
            # matmul on the device.
            with torch.no_grad():
                junk = (torch.ones(64, 16).to(live_device) @ torch.ones(16, 64).to(live_device)).cpu()
            logger.info("[neuron] EXPERIMENT dummy device op done (sum %.1f)", float(junk.sum()))
        try:
            touched, nbytes, t_delta, t_copy = apply(where == "device")
        except NotImplementedError:
            raise
        except Exception as e:  # noqa: BLE001
            if where == "host":
                raise
            logger.warning("[neuron] EXPERIMENT adapter update on the device failed (%s: %s); from the host from now on",
                           type(e).__name__, str(e)[:300])
            state["on_device"] = False
            base_dev.clear()
            where = "host (device path failed)"
            touched, nbytes, t_delta, t_copy = apply(False)
        logger.info("[neuron] EXPERIMENT adapter update from %s on the %s (write " + write_mode + "): adapter %d bytes, %d live tensors (%d bytes) "
                    "rewritten, delta %.2fs, copy %.2fs, total %.2fs", path, where,
                    sum(t.numel() * t.element_size() for t in raw.values()), touched, nbytes, t_delta, t_copy,
                    time.time() - t0)

    def _set_stage(self, config) -> None:
        """The pipeline stage this host builds (distributed.pipeline.current_stage_spec), or the whole model without
        PP."""
        from sglang_neuron.distributed.pipeline import current_stage_spec

        self.stage = current_stage_spec(int(config.num_hidden_layers))
        if self.stage is None:
            self.start_layer, self.end_layer = 0, int(config.num_hidden_layers)
            self.is_first_stage = self.is_last_stage = True
            return
        self.start_layer, self.end_layer = self.stage.start_layer, self.stage.end_layer
        self.is_first_stage, self.is_last_stage = self.stage.is_first, self.stage.is_last

    def check_launch(self, config) -> None:
        """Refuse, at construction, a launch this model's graphs cannot serve."""

    def build_graph(self, phase: str, config) -> nn.Module:
        raise NotImplementedError

    def neuron_compiler_args(self, phase: str, default_args: list[str]) -> list[str]:
        """The decode compiles at -O2, without the modular-flow threshold, unless it traces the device sampler's top-k
        (top logprobs, or topk_topp), which -O2 miscompiles; a router's top-k is unaffected. Prefill keeps defaults."""
        if phase != "decode" or sampling.MAX_TOP_LOGPROBS > 0 or sampling.sampler_mode() == "topk_topp":
            return default_args
        prefix = "--internal-hlo2tensorizer-options="
        args = []
        for arg in default_args:
            if arg in ("-O0", "-O1", "-O3"):
                arg = "-O2"
            elif arg.startswith(prefix):
                options = [o for o in shlex.split(arg[len(prefix):])
                           if not o.startswith("--modular-flow-mac-threshold=")]
                if not options:
                    continue
                arg = prefix + shlex.join(options)
            args.append(arg)
        return args

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """SGLang hands over a lazy iterator over the checkpoint, left unread: the first graph reads this rank's shards
        of the checkpoint, and an aggregated server's other graph shares its device copy."""
        if self._weights_loaded:
            raise NotImplementedError(f"{self.name} on Neuron does not update its weights in place")
        path = self.hf_config._name_or_path
        device = neuron_device()
        (first_phase, first), *others = self.model_graphs.items()
        first.load_weights(path, device)
        place_rest_on_device(self.name, first_phase, first, device)
        # an aggregated server's graphs declare the same parameters, so the second shares the first's device copy
        for phase, graph in others:
            share_weights(first, graph, name=self.name, phase=phase, source_phase=first_phase)
            place_rest_on_device(self.name, phase, graph, device)
        self._weights_loaded = True
        logger.info("%s weights loaded onto %s from %s", self.name, device, path)
        directory = envs.weight_hash_dir()
        if directory:
            write_weight_hashes(self.model_graphs, sglang_tree.rank_label(sglang_tree.parallel_config()), directory)


def write_weight_hashes(graphs: dict, rank, directory: str) -> None:
    """SGLANG_NEURON_WEIGHT_HASH_DIR: write the md5 of every loaded parameter and buffer, to find tensors that differ
    between ranks or launches. One file per rank and graph, weights_rank<rank>_<phase>.txt (rank labelled as in the
    trace files), one sorted line per tensor: kind, name, shape, dtype, md5 ("-" if empty). Slow: it copies every
    tensor to the host."""
    import hashlib
    import os

    os.makedirs(directory, exist_ok=True)
    for phase, graph in graphs.items():
        lines = []
        for kind, tensors in (("param", graph.named_parameters()), ("buffer", graph.named_buffers())):
            for name, tensor in tensors:
                data = tensor.detach().to("cpu").contiguous().reshape(-1)
                # md5 reads the host copy's buffer in place
                digest = hashlib.md5(data.view(torch.uint8).numpy(), usedforsecurity=False).hexdigest() \
                    if data.numel() else "-"
                lines.append(f"{kind} {name} {tuple(tensor.shape)} {tensor.dtype} {digest}\n")
        path = os.path.join(directory, f"weights_rank{rank}_{phase}.txt")
        with open(path, "w") as f:
            f.writelines(sorted(lines))
        logger.info("[neuron] the md5 of the %s graph's %d tensors written to %s", phase, len(lines), path)


def place_rest_on_device(name: str, phase: str, graph: nn.Module, device) -> None:
    """Move what the checkpoint did not provide (rotary buffers, unloaded parameters) to ``device``."""
    for buffer_name, buf in graph.named_buffers():
        if buf.device.type != device.type:
            owner, _, attr = buffer_name.rpartition(".")
            graph.get_submodule(owner)._buffers[attr] = buf.to(device)
    moved = []
    for param_name, param in graph.named_parameters():
        if param.device.type != device.type:
            owner, _, attr = param_name.rpartition(".")
            graph.get_submodule(owner)._parameters[attr] = nn.Parameter(param.detach().to(device),
                                                                        requires_grad=param.requires_grad)
            moved.append(param_name)
    if moved:
        logger.info("%s %s: %d parameter(s) not in the checkpoint moved to the device: %s",
                    name, phase, len(moved), moved[:8])


# KV-cache quantization scales a layer's attention holds outside its parameters
_KV_SCALES = ("k_scale", "v_scale", "k_scale_float", "v_scale_float")


def share_weights(source: nn.Module, graph: nn.Module, *, name: str, phase: str, source_phase: str) -> None:
    """Give ``graph`` the parameters and KV scales of the loaded ``source`` graph. A parameter the two do not share
    (by name, shape and dtype) is refused rather than loaded twice."""
    # every name, a tied parameter's too, so the second graph keeps the first's ties
    params = dict(source.named_parameters(remove_duplicate=False))
    unlike = [n for n, p in graph.named_parameters(remove_duplicate=False)
              if n not in params or params[n].shape != p.shape or params[n].dtype != p.dtype]
    if unlike:
        raise RuntimeError(f"the {name} {phase} graph has {len(unlike)} parameter(s) the {source_phase} graph does not "
                           f"share, so they cannot share one device copy of the weights: {unlike[:8]}")
    for n, _ in list(graph.named_parameters(remove_duplicate=False)):
        owner, _, attr = n.rpartition(".")
        graph.get_submodule(owner)._parameters[attr] = params[n]
    for module_name, src in source.named_modules():
        for attr in _KV_SCALES:
            if hasattr(src, attr):
                setattr(graph.get_submodule(module_name), attr, getattr(src, attr))
    logger.info("%s %s graph shares the %s graph's %d device parameters", name, phase, source_phase, len(params))


class SampledExecution(nn.Module):
    """A role graph followed by the device sampler. The graph's logits are this rank's vocabulary shard of its own
    sampled rows, so the per-row sampling inputs need no gather."""

    sampler_modes = sampling.SAMPLER_MODES

    def __init__(self, graph: nn.Module, mode: str):
        super().__init__()
        self.model_graph = graph
        self.group = graph.lm_head_tp_group
        self.vocabulary_size = graph.config.vocab_size
        self.mode = mode

    def forward(self, **kwargs):
        rank = self.group.rank_in_group_tensor
        if self.mode == "greedy":
            return sampling.greedy_sharded_logits(self.model_graph(**kwargs), self.group, rank, self.vocabulary_size)
        temperatures = kwargs.pop("sampling_temperatures")
        seeds = kwargs.pop("sampling_seeds")
        filters = kwargs.pop("sampling_filters", None)
        logits = self.model_graph(**kwargs)
        keys = sampling.noise_row_keys(seeds, sampling.sampled_positions(kwargs["positions"],
                                                                         kwargs["sampling_positions"]))
        if self.mode == "topk_topp":
            return sampling.sample_filtered_sharded_logits(logits, self.group, rank, self.vocabulary_size,
                                                           temperatures, keys, filters)
        return sampling.sample_sharded_logits(logits, self.group, rank, self.vocabulary_size, temperatures, keys)


def sampled_execution(graph: nn.Module, mode: str) -> nn.Module:
    """What runs a role graph's step: the graph itself where it samples in its own forward (``samples_in_forward``,
    the GPT-OSS giga-kernel decode's fused tail), else the graph followed by the device sampler."""
    return graph if getattr(graph, "samples_in_forward", False) else SampledExecution(graph, mode)
