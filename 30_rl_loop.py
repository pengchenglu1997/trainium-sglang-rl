#!/usr/bin/env python3
"""A small RL loop on GSM8K that times every phase of a step.

    python 30_rl_loop.py --run-tag sgl --backend sglang_neuron --backends-json backends.json \
        --base-model DIR --train-jsonl gsm8k_train.jsonl --eval-jsonl gsm8k_test.jsonl \
        --device neuron --core 8 --results DIR --work-dir DIR [--steps 20]

One step: sample K answers for each of Q questions (rollout) -> reward (right final number = 1)
-> group advantages -> one policy-gradient update of the LoRA (trainer.py) -> every --sync-every
steps the merged weights are written and the server is restarted on them (the only way to give a
server new weights when it has no weight-update call). Backend "hf" generates inside the trainer
process instead (model.generate), so it has no server and no sync.

The step line holds the seconds of each phase: rollout, reward, train, export, restart (stop and
start until the first answer). Greedy accuracy on a fixed held-out list is taken before the first
step and after the last one. Counts and timings only; no question text is written.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rl_common as C  # noqa: E402
import rollout_client as rc  # noqa: E402
import server_ctl  # noqa: E402


def evaluate(a, rows, servers, trainer):
    """Greedy accuracy on rows. Returns (accuracy, seconds, errors). servers: the replicas (all on
    the same weights), questions spread over them; None = the trainer generates."""
    t0 = time.time()
    errors = 0
    if servers:
        def one(ir):
            i, r = ir
            s = servers[i % len(servers)]
            x = rc.chat(s.base_url, s.served_name(), C.messages_for(r[1]),
                        temperature=0.0, max_tokens=a.eval_max_new_tokens, timeout=a.timeout)
            return (x["texts"][0] if x["status"] == 200 and x["texts"] else None)
        with ThreadPoolExecutor(max_workers=a.eval_concurrency * len(servers)) as ex:
            outs = list(ex.map(one, list(enumerate(rows))))
    else:
        outs = []
        for r in rows:
            texts, _ = trainer.generate(r[1], 1, 0.0, a.eval_max_new_tokens)
            outs.append(texts[0])
    errors = sum(o is None for o in outs)
    ok = sum(C.is_correct(o, r[2]) for o, r in zip(outs, rows) if o is not None)
    if rows and errors > 0.05 * len(rows):
        raise RuntimeError(f"evaluation: {errors} of {len(rows)} requests failed")
    # accuracy over the questions that were answered; the error count is reported next to it
    answered = len(rows) - errors
    return round(ok / answered, 4) if answered else None, round(time.time() - t0, 1), errors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-tag", required=True)
    ap.add_argument("--backend", required=True, help="a name in backends.json, or 'hf'")
    ap.add_argument("--backends-json", default=str(Path(__file__).with_name("backends.json")))
    ap.add_argument("--base-model", required=True, help="local model directory")
    ap.add_argument("--train-jsonl", required=True)
    ap.add_argument("--eval-jsonl", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--work-dir", required=True, help="merged checkpoints and server logs")
    ap.add_argument("--device", choices=["neuron", "cuda", "cpu"], required=True)
    ap.add_argument("--core", default=None, help="NeuronCore of the trainer (neuron only)")
    ap.add_argument("--port", type=int, default=30000)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--questions-per-step", type=int, default=4)
    ap.add_argument("--k", type=int, default=8, help="sampled answers per question")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--max-prompt", type=int, default=320)
    ap.add_argument("--micro-bs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4,
                    help="LoRA learning rate. The server computes in bf16: with a much smaller "
                         "rate the merged weights round back to the base (the step line gives "
                         "the share of merged weights that changed)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--sync-every", type=int, default=1)
    ap.add_argument("--sync-method", choices=["auto", "restart", "sglang_disk"], default="auto",
                    help="auto: the backend's own (restart unless backends.json says otherwise)")
    ap.add_argument("--ckpt-dir", default=None,
                    help="where merged checkpoints go (must be under MODELS_ROOT for a "
                         "containerised server); default the work dir")
    ap.add_argument("--rollout-mode", choices=["auto", "n", "parallel"], default="parallel",
                    help="parallel: K seeded requests at once (what SGLang-Neuron documents; it "
                         "refuses a seed together with n > 1)")
    ap.add_argument("--rollout-concurrency", type=int, default=4, help="questions at once")
    ap.add_argument("--eval-n", type=int, default=100)
    ap.add_argument("--eval-max-new-tokens", type=int, default=512)
    ap.add_argument("--eval-concurrency", type=int, default=16)
    ap.add_argument("--timeout", type=float, default=900)
    ap.add_argument("--keep-checkpoints", action="store_true")
    ap.add_argument("--replicas", type=int, default=1,
                    help="rollout servers on the same weights; --rollout-concurrency is per replica")
    ap.add_argument("--replica-cores", default=None,
                    help="one NeuronCore range per replica, separated by ';' (0-3;4-7;...): passed "
                         "to each server as SGL_CORES")
    ap.add_argument("--sync-payload", choices=["merged", "adapter"], default="merged",
                    help="adapter: only the LoRA adapter is written and sent (a server that takes "
                         "an adapter folder in update_weights_from_disk; in-place sync only)")
    ap.add_argument("--train-compile", action="store_true",
                    help="training forward under torch.compile (Neuron)")
    ap.add_argument("--overlap", action="store_true",
                    help="sample the next step's answers while this step trains (one update behind)")
    ap.add_argument("--len-buckets", default=None,
                    help="training sequences padded to the smallest of these total lengths (ascending, "
                         "last = max-prompt + max-new-tokens), one compiled shape each")
    ap.add_argument("--answer-window", action="store_true",
                    help="training: vocabulary logits only at the answer positions")
    ap.add_argument("--eval-every", type=int, default=0,
                    help="also evaluate after every N-th step (after its sync); 0 = only before and after")
    ap.add_argument("--train-cores", default=None,
                    help="extra NeuronCores for training, comma-separated (one worker process each, "
                         "next to the trainer on --core): ptrainer.py")
    ap.add_argument("--max-limit-share", type=float, default=None,
                    help="stop training when this share of the sampled answers ran into the "
                         "token limit in two steps in a row (answers growing without end)")
    a = ap.parse_args()

    res = C.Results(a.results, f"30_rl_loop_{a.run_tag}")
    work = Path(a.work_dir) / a.run_tag
    work.mkdir(parents=True, exist_ok=True)
    server = None
    servers = []
    try:
        import trainer as T
        if a.device == "neuron":
            T.set_neuron_env(a.core, compile_train=a.train_compile)
        t0 = time.time()
        tkw = dict(lr=a.lr, max_prompt=a.max_prompt, max_completion=a.max_new_tokens,
                   micro_bs=a.micro_bs, seed=a.seed, compile_train=a.train_compile)
        if a.len_buckets:
            tkw["len_buckets"] = [int(x) for x in a.len_buckets.split(",")]
        if a.answer_window:
            tkw["answer_window"] = True
        if a.train_cores:
            if a.device != "neuron":
                raise RuntimeError("--train-cores is for the neuron device")
            import ptrainer
            tr = ptrainer.ParallelTrainer(a.base_model, worker_cores=a.train_cores.split(","), **tkw)
        else:
            tr = T.Trainer(a.base_model, device=a.device, **tkw)
        res.write(event="trainer_ready", seconds=round(time.time() - t0, 1), device=a.device,
                  core=a.core, trainable_params=sum(p.numel() for p in tr.params),
                  max_len=tr.max_len, micro_bs=a.micro_bs, lr=a.lr, **tr.sync_check())

        train = C.load_gsm8k(a.train_jsonl)
        keep = [r for r in train if tr.prompt_tokens(r[1]) <= a.max_prompt]
        need = a.steps * a.questions_per_step
        evalrows = C.load_gsm8k(a.eval_jsonl, limit=a.eval_n)
        res.write(event="data", train_rows=len(train), train_rows_within_prompt_limit=len(keep),
                  needed=need, eval_rows=len(evalrows))
        if len(keep) < need:
            raise RuntimeError(f"only {len(keep)} training questions fit the prompt limit, {need} needed")

        res.write(event="start", run_tag=a.run_tag, backend=a.backend, steps=a.steps,
                  questions_per_step=a.questions_per_step, k=a.k, temperature=a.temperature,
                  max_new_tokens=a.max_new_tokens, sync_every=a.sync_every,
                  rollout_mode=a.rollout_mode, seed=a.seed, replicas=a.replicas,
                  replica_cores=a.replica_cores, sync_payload=a.sync_payload,
                  train_compile=a.train_compile, lr=a.lr, max_limit_share=a.max_limit_share)

        sync_method = "restart"
        ckpt_root = Path(a.ckpt_dir) / a.run_tag if a.ckpt_dir else work
        ckpt_root.mkdir(parents=True, exist_ok=True)
        if a.backend != "hf":
            backend = server_ctl.load_backend(a.backends_json, a.backend)
            if a.sync_method == "auto":
                sync_method = backend.get("weight_update", "restart")
            else:
                sync_method = a.sync_method
            if a.sync_payload == "adapter" and sync_method != "sglang_disk":
                raise RuntimeError("--sync-payload adapter needs the in-place sync (sglang_disk)")
            cores = a.replica_cores.split(";") if a.replica_cores else [None] * a.replicas
            if len(cores) != a.replicas:
                raise RuntimeError(f"--replica-cores names {len(cores)} ranges for {a.replicas} replicas")
            for i in range(a.replicas):
                servers.append(server_ctl.Server(
                    backend, a.port + i, work / "server_logs",
                    name=f"rl_{a.run_tag}" if a.replicas == 1 else f"rl_{a.run_tag}_{i}",
                    env={"SGL_CORES": cores[i]} if cores[i] else None))
            server = servers[0]
            res.write(event="sync", method=sync_method, ckpt_root=str(ckpt_root))
            # the first replica alone (it fills a cold compile cache), then the others together
            st = server.start(a.base_model)
            res.write(event="server_start", replica=0, **st)
            if st["ready_s"] is None:
                raise RuntimeError(f"server did not become ready on the base model ({st})")
            if len(servers) > 1:
                t = time.time()
                with ThreadPoolExecutor(max_workers=len(servers) - 1) as ex:
                    sts = list(ex.map(lambda s: s.start(a.base_model), servers[1:]))
                for i, st in enumerate(sts, start=1):
                    res.write(event="server_start", replica=i, **st)
                    if st["ready_s"] is None:
                        raise RuntimeError(f"replica {i} did not become ready ({st})")
                res.write(event="replicas_ready", replicas=len(servers), others_s=round(time.time() - t, 1))

        acc, sec, err = evaluate(a, evalrows, servers, tr)
        res.write(event="eval", when="before", greedy_accuracy=acc, seconds=sec, errors=err,
                  n=len(evalrows))

        prev_dir = None
        mode_used = None
        limit_streak = 0
        stopped_early = None
        pending = False
        prefetch = None
        overlap_pool = ThreadPoolExecutor(max_workers=1)
        t_loop = time.time()
        t_prev = t_loop
        small_files = [f for f in Path(a.base_model).iterdir() if f.is_file() and f.suffix != ".safetensors"]
        for step in range(1, a.steps + 1):
            qs = keep[(step - 1) * a.questions_per_step: step * a.questions_per_step]
            # rollout
            t = time.time()
            n_tok = 0
            roll_err = 0
            first_error = None
            overlapped = False
            if server is not None:
                def server_rollout(st, rows_, mode_):
                    def one(pr):
                        qpos, r = pr
                        # seeds of different questions and steps never overlap (K seeds per group)
                        s = servers[qpos % len(servers)]
                        return rc.sample_group(s.base_url, s.served_name(),
                                               C.messages_for(r[1]), a.k, a.temperature,
                                               a.max_new_tokens, mode=mode_,
                                               seed=a.seed * 10 ** 7 + st * 10 ** 4 + qpos * a.k,
                                               timeout=a.timeout)
                    t_ = time.time()
                    with ThreadPoolExecutor(max_workers=a.rollout_concurrency * len(servers)) as ex:
                        out_ = list(ex.map(one, list(enumerate(rows_))))
                    return out_, time.time() - t_
                if prefetch is not None:
                    # sampled while the previous step trained: from the weights of one sync earlier
                    groups, pre_s = prefetch.result()
                    prefetch = None
                    overlapped = True
                else:
                    groups, pre_s = server_rollout(step, qs, mode_used or a.rollout_mode)
                mode_used = groups[0]["mode"]
                texts = [g["texts"] if len(g["texts"]) == a.k else [None] * a.k for g in groups]
                n_tok = sum(g["completion_tokens"] or 0 for g in groups)
                roll_err = sum(x is None for tx in texts for x in tx)
                first_error = next((g["first_error"] for g in groups if g["first_error"]), None)
                nrep = len(servers)
                # texts[j::nrep]: the groups replica j served (questions go round-robin)
                dead = [j for j in range(nrep) if texts[j::nrep]
                        and all(x is None for tx in texts[j::nrep] for x in tx)]
                if roll_err > 0.5 * len(qs) * a.k or (nrep > 1 and dead):
                    raise RuntimeError(f"rollout failed at step {step}: {roll_err} of "
                                       f"{len(qs) * a.k} answers missing (replicas with no answer: "
                                       f"{dead}); first error: {first_error}")
            else:
                texts = []
                for r in qs:
                    tx, nt = tr.generate(r[1], a.k, a.temperature, a.max_new_tokens)
                    texts.append(tx)
                    n_tok += nt
                mode_used = "hf_generate"
            roll_s = pre_s if server is not None else time.time() - t
            # reward and advantages
            t = time.time()
            items, rewards, zero_groups = [], [], 0
            for r, tx in zip(qs, texts):
                valid = [x for x in tx if x is not None]      # a failed request is no answer
                rs = [C.reward(x, r[2]) for x in valid]
                adv = C.group_advantages(rs) if len(valid) >= 2 else [0.0] * len(valid)
                zero_groups += int(all(v == 0.0 for v in adv))
                rewards += rs
                items += [(r[1], x, v) for x, v in zip(valid, adv)]
            rew_s = time.time() - t
            if not rewards:
                raise RuntimeError(f"no answers at step {step}")
            # answers that ran into the token limit (the model no longer stops by itself)
            answered = [x for tx in texts for x in tx if x is not None]
            at_limit = sum(len(tr.tok(x, add_special_tokens=False)["input_ids"]) >= a.max_new_tokens - 1
                           for x in answered)
            limit_share = round(at_limit / len(answered), 4)
            # --overlap: the next step's answers are sampled while this step trains (they come from the
            # weights now on the servers, one update behind the trainer). The sync below then waits for
            # them, because SGLang's update takes its lock only when no request is in flight.
            if a.overlap and server is not None and step < a.steps:
                nxt = keep[step * a.questions_per_step: (step + 1) * a.questions_per_step]
                prefetch = overlap_pool.submit(server_rollout, step + 1, nxt, mode_used or a.rollout_mode)
            # update
            upd = tr.update(items, temperature=a.temperature)
            tr.free()
            # new weights to the server
            exp_s = restart = inplace = merged_changed = None
            pending = pending or not upd.get("skipped")     # trained weights not yet on the servers
            if a.max_limit_share is not None:
                limit_streak = limit_streak + 1 if limit_share >= a.max_limit_share else 0
            last = step == a.steps or (a.max_limit_share is not None and limit_streak >= 2)
            # always before the final evaluation, whatever --sync-every says
            if server is not None and pending and (step % a.sync_every == 0 or last):
                pending = False
                if a.sync_payload == "adapter":
                    ck = ckpt_root / f"adapter_step{step:04d}"
                    t = time.time()
                    tr.save_adapter(ck)
                    for f in small_files:       # SGLang reads the config and tokenizer from the path
                        shutil.copy2(f, ck / f.name)
                    exp_s = round(time.time() - t, 3)
                else:
                    ck = ckpt_root / f"merged_step{step:04d}"
                    exp = tr.save_merged(ck)
                    exp_s = exp["seconds"]
                    merged_changed = exp["share_of_merged_weights_changed"]
                if sync_method == "sglang_disk":
                    t = time.time()
                    with ThreadPoolExecutor(max_workers=len(servers)) as ex:
                        ups = list(ex.map(lambda s: s.update_weights_from_disk(ck), servers))
                    inplace = next((u for u in ups if not u["ok"]), None) or dict(
                        ups[0], seconds=round(time.time() - t, 2), per_replica_s=[u["seconds"] for u in ups])
                    if not inplace["ok"]:
                        res.write(event="step", step=step, inplace=inplace,
                                  status_note="weight update refused")
                        raise RuntimeError(f"update_weights_from_disk failed at step {step}")
                else:
                    if len(servers) > 1:
                        raise RuntimeError("restart-based sync is written for one server only")
                    restart = server.restart(ck)
                    if restart["ready_s"] is None:
                        res.write(event="step", step=step, restart=restart,
                                  status_note="restart failed")
                        raise RuntimeError(f"server did not become ready on the weights of step {step}")
                if prev_dir is not None and not a.keep_checkpoints:
                    shutil.rmtree(prev_dir, ignore_errors=True)
                prev_dir = ck
            res.write(event="step", step=step, questions=len(qs), answers=len(rewards),
                      rollout_mode=mode_used, rollout_s=round(roll_s, 2),
                      rollout_errors=roll_err, rollout_first_error=first_error,
                      completion_tokens=n_tok or None,
                      reward_s=round(rew_s, 3), reward_mean=round(sum(rewards) / len(rewards), 4),
                      groups_without_signal=zero_groups,
                      answers_at_token_limit=at_limit, share_at_token_limit=limit_share,
                      rollout_overlapped=overlapped, step_wall_s=round(time.time() - t_prev, 2),
                      train_s=None if upd.get("skipped") else upd["seconds"],
                      train_sequences=upd["sequences"], train_skipped=upd.get("skipped"),
                      micro_batches=upd["micro_batches"],
                      train_processes=upd.get("processes"), train_send_s=upd.get("send_s"),
                      train_own_share_s=upd.get("own_share_s"),
                      train_slowest_worker_s=upd.get("slowest_worker_s"),
                      share_of_merged_weights_changed=merged_changed,
                      loss=upd["loss"], mean_token_logp=upd["mean_token_logp"],
                      export_s=exp_s, sync_method=sync_method if server is not None else None,
                      inplace_update_s=None if inplace is None else inplace["seconds"],
                      restart_stop_s=None if restart is None else restart["stop_s"],
                      restart_ready_s=None if restart is None else restart["ready_s"],
                      trainer_peak_bytes=tr.peak_bytes())
            print(f"step {step}/{a.steps} reward {sum(rewards) / len(rewards):.3f} "
                  f"rollout {roll_s:.1f}s train {upd['seconds']}s restart "
                  f"{None if restart is None else restart['ready_s']}", flush=True)
            t_prev = time.time()
            if a.eval_every and step % a.eval_every == 0 and step != a.steps and not pending:
                acc_m, sec_m, err_m = evaluate(a, evalrows, servers, tr)
                res.write(event="eval", when=f"step {step}", step=step, greedy_accuracy=acc_m, seconds=sec_m,
                          errors=err_m, n=len(evalrows))
                t_prev = time.time()        # the evaluation is not part of the next step's wall time
            if a.max_limit_share is not None:
                if limit_streak >= 2:
                    stopped_early = step
                    res.write(event="stopped_early", step=step, share_at_token_limit=limit_share,
                              reason="answers at the token limit in two steps in a row")
                    break

        loop_wall = time.time() - t_loop
        if prefetch is not None:        # an early stop leaves one sampled batch unused
            try:
                prefetch.result()
            except Exception:  # noqa: BLE001
                pass
        acc2, sec2, err2 = evaluate(a, evalrows, servers, tr)
        res.write(event="eval", when="after", greedy_accuracy=acc2, seconds=sec2, errors=err2,
                  n=len(evalrows))
        tr.save_adapter(work / "final_adapter")
        steps = [r for r in C.read_jsonl(res.path) if r.get("event") == "step"]

        def col(k):
            return C.describe([s[k] for s in steps[1:] if s.get(k) is not None])

        res.write(event="summary", run_tag=a.run_tag, backend=a.backend, steps=len(steps),
                  rollout_mode=mode_used, eval_before=acc, eval_after=acc2,
                  rollout_s=col("rollout_s"), train_s=col("train_s"), export_s=col("export_s"),
                  restart_stop_s=col("restart_stop_s"), restart_ready_s=col("restart_ready_s"),
                  inplace_update_s=col("inplace_update_s"), sync_method=sync_method,
                  reward_mean_first5=round(sum(s["reward_mean"] for s in steps[:5]) / max(1, len(steps[:5])), 4),
                  reward_mean_last5=round(sum(s["reward_mean"] for s in steps[-5:]) / max(1, len(steps[-5:])), 4),
                  rollout_errors=sum(s["rollout_errors"] for s in steps),
                  steps_with_update=sum(1 for s in steps if not s["train_skipped"]),
                  replicas=a.replicas, sync_payload=a.sync_payload, train_compile=a.train_compile,
                  stopped_early_at=stopped_early, overlap=a.overlap,
                  loop_wall_s=round(loop_wall, 1),
                  step_wall_s=C.describe([s["step_wall_s"] for s in steps[1:] if s.get("step_wall_s") is not None]),
                  step_s=C.describe([round(sum(s.get(k) or 0 for k in
                                               ("rollout_s", "reward_s", "train_s", "export_s", "inplace_update_s",
                                                "restart_stop_s", "restart_ready_s")), 2) for s in steps[1:]]),
                  share_at_token_limit_max=max((s.get("share_at_token_limit") or 0) for s in steps),
                  note="phase statistics leave out step 1 (first shapes are compiled there); "
                       "train_s only over steps that made an update")
    except BaseException as e:  # noqa: BLE001  (also a kill signal: the file must not look finished)
        res.fail("loop", e)
        raise
    finally:
        for s in servers:
            try:
                s.stop()
            except Exception:  # noqa: BLE001
                pass
        try:
            closer = getattr(locals().get("tr"), "close", None)
            if callable(closer):
                closer()
        except Exception:  # noqa: BLE001
            pass
        res.done()


if __name__ == "__main__":
    main()
