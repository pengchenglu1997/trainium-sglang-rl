#!/usr/bin/env python3
"""W0.1: how much of the fixed 576-position training length is padding, from saved RL-loop runs.

    python 63_padding_share.py RESULT_DIR [RESULT_DIR ...] [--runs long1,plain_s2,plain_s3,plain_s4]
        [--assume-prompt 60,100,140] [--out FILE]

Reads 30_rl_loop_<run>.jsonl in every folder given (first folder that holds a run wins). The trainer
(trainer.py) pads every training sequence (prompt + answer + end token) to max_len = 576 unless
--len-buckets is set; only answers with a non-zero advantage are trained.

Two cases, decided per run from the step lines:
  A. the step lines hold "token_counts" (30_rl_loop.py after 2026-11-22: sorted answer tokens of every
     answer and sorted full lengths of the trained rows). Then the script prints the length
     distribution, the padding share at 576 and with the length buckets (256,384,576) and
     (192,256,384,512,576), and the share of answers at the 256-token limit.
  B. the step lines hold only completion_tokens (sum over the answers of the step), answers and
     answers_at_token_limit (all runs saved before that change). No prompt length and no
     per-answer length is saved, so the script computes only what the sums fix: mean answer length,
     share at the limit, and the padding share at 576 as a function of the prompt length P
     (an upper bound at P = 0, and an assumed-P sweep that is marked ASSUMED). When the SGLang
     server logs of the run were saved (<folder>/../work_<run>/<run>/server_logs), the prompt length
     of every request is read from their prefill lines and the share is also given for that measured
     mean prompt. Bucket shares need the joint distribution and are not computed in this case.
Padded positions follow Trainer.accumulate: micro-batches of micro_bs rows (from the trainer_ready
line), empty rows of a part-filled micro-batch count as padding.
Checks nothing about the model; counts only. Writes the table to --out as well as to the screen.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rl_common as C  # noqa: E402

MAX_LEN = 576
BUCKETS = [(256, 384, 576), (192, 256, 384, 512, 576)]
DEFAULT_OUT = "63_padding_share.txt"


def find_run(folders, run):
    for d in folders:
        p = Path(d) / f"30_rl_loop_{run}.jsonl"
        if p.exists():
            return p
    return None


def pct(x):
    return "n/a" if x is None else f"{100 * x:.1f}%"


def quantile(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


def pad_share(step_lens, micro_bs, buckets=None):
    """1 - real positions / padded positions over steps (one list of row lengths per step).
    buckets None: every row padded to MAX_LEN. As Trainer.accumulate does, rows go in micro-batches
    of micro_bs rows per bucket, and the empty rows of a part-filled micro-batch count as padding.
    (ParallelTrainer buckets each process's share separately, so with several processes and
    buckets the real share is somewhat higher than this.)"""
    bs = [MAX_LEN] if buckets is None else list(buckets)
    real = padded = 0
    for lens in step_lens:
        real += sum(lens)
        counts = {}
        for n in lens:
            b = next(b for b in bs if n <= b)
            counts[b] = counts.get(b, 0) + 1
        padded += sum(-(-c // micro_bs) * micro_bs * b for b, c in counts.items())
    return 1 - real / padded if padded else None


PREFILL = re.compile(r"Prefill batch, #new-seq: (\d+), #new-token: (\d+), #cached-token: (\d+)")


def prompt_lens_from_logs(folder, run):
    """Prompt length of every request from the SGLang server logs saved next to the run
    (<folder>/../work_<run>/<run>/server_logs/*.log): #new-token + #cached-token of each prefill
    line with one new sequence. Training and evaluation requests both appear; lines under 40
    tokens (start-up probes) are dropped. None when no log is there."""
    d = Path(folder).parent / f"work_{run}" / run / "server_logs"
    files = sorted(d.glob("*.log")) if d.is_dir() else []
    out, multi = [], 0
    for f in files:
        for line in f.open(encoding="utf-8", errors="replace"):
            m = PREFILL.search(line)
            if m:
                s, n, c = map(int, m.groups())
                if s == 1:
                    if n + c >= 40:
                        out.append(n + c)
                else:
                    multi += 1
    return (out, multi, len(files)) if out else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folders", nargs="+")
    ap.add_argument("--runs", default="long1,plain_s2,plain_s3,plain_s4")
    ap.add_argument("--assume-prompt", default="60,100,140",
                    help="prompt lengths (tokens) for the ASSUMED sweep in case B; not measured")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    a = ap.parse_args()
    lines = []

    def emit(s=""):
        lines.append(s)
        print(s)

    assume = [int(x) for x in a.assume_prompt.split(",") if x]
    emit("W0.1 padding share of the training length (max_len %d); folders: %s" % (MAX_LEN, "; ".join(a.folders)))
    pooled = {"n_ans": 0, "tok": 0, "lim": 0, "seq": [], "plog": []}
    exact_all = True
    rows_b = []
    for run in a.runs.split(","):
        p = find_run(a.folders, run)
        if p is None:
            emit(f"run {run}: no 30_rl_loop_{run}.jsonl in the folders given")
            exact_all = False
            continue
        recs = C.read_jsonl(p)
        steps = [r for r in recs if r.get("event") == "step"]
        finished = bool(recs) and recs[-1] == {"done": True}
        start = next((r for r in recs if r.get("event") == "start"), {})
        ready = next((r for r in recs if r.get("event") == "trainer_ready"), {})
        micro_bs = ready.get("micro_bs") or 4
        mnt = start.get("max_new_tokens", a.max_new_tokens)
        n_ans = sum(s["answers"] for s in steps)
        tok = sum(s.get("completion_tokens") or 0 for s in steps)
        lim = sum(s.get("answers_at_token_limit") or 0 for s in steps)
        trained = sum(s.get("train_sequences") or 0 for s in steps)
        has_tc = bool(steps) and all("token_counts" in s for s in steps)
        pooled["n_ans"] += n_ans
        pooled["tok"] += tok
        pooled["lim"] += lim
        mean_a = tok / n_ans
        per_step = [s["completion_tokens"] / s["answers"] for s in steps if s.get("completion_tokens")]
        lim_share_steps = [s["share_at_token_limit"] for s in steps if s.get("share_at_token_limit") is not None]
        emit()
        emit(f"== {run}  ({p.name}; {len(steps)} steps, finished={finished}, max_new_tokens {mnt}, "
             f"questions/step {start.get('questions_per_step')}, k {start.get('k')})")
        emit(f"   saved fields: " + ("token_counts (per-answer lengths)" if has_tc else
             "answers, completion_tokens (sum per step), answers_at_token_limit, train_sequences; "
             "no prompt tokens and no per-answer lengths in the jsonl"))
        emit(f"   answers {n_ans}, trained rows {trained} ({pct(trained / n_ans)} of answers, "
             f"zero-advantage rows are left out)")
        emit(f"   mean answer tokens (all answers) {mean_a:.1f}; per-step mean: min {min(per_step):.1f} "
             f"median {quantile(per_step, 0.5):.1f} max {max(per_step):.1f}; first 10 steps "
             f"{sum(per_step[:10]) / len(per_step[:10]):.1f}, last 10 steps {sum(per_step[-10:]) / len(per_step[-10:]):.1f}")
        emit(f"   answers at the {mnt}-token limit: {lim} of {n_ans} = {pct(lim / n_ans)}; per-step share: "
             f"min {pct(min(lim_share_steps))} median {pct(quantile(lim_share_steps, 0.5))} "
             f"max {pct(max(lim_share_steps))}")
        if has_tc:
            seq_steps = [s["token_counts"]["trained_seq"] for s in steps]
            seq = [n for x in seq_steps for n in x]
            ans = [n for s in steps for n in s["token_counts"]["answers"]]
            pooled["seq"] += seq_steps
            emit(f"   sequence length of trained rows (prompt + answer + end): n {len(seq)} min {min(seq)} "
                 f"p25 {quantile(seq, .25)} median {quantile(seq, .5)} p75 {quantile(seq, .75)} "
                 f"p95 {quantile(seq, .95)} max {max(seq)}")
            emit(f"   answer tokens per answer: median {quantile(ans, .5)} p95 {quantile(ans, .95)} max {max(ans)}")
            emit(f"   padding share at {MAX_LEN} (micro-batches of {micro_bs}): {pct(pad_share(seq_steps, micro_bs))}")
            for b in BUCKETS:
                emit(f"   padding share with buckets {b}: {pct(pad_share(seq_steps, micro_bs, b))}")
        else:
            exact_all = False
            emit(f"   padding share at {MAX_LEN}, from the sums (mean length over all answers stands in for the "
                 f"trained rows; end token not added):")
            emit(f"     upper bound (prompt length 0): {pct(1 - mean_a / MAX_LEN)}")
            for P in assume:
                emit(f"     ASSUMED prompt length {P}: {pct(1 - (P + mean_a) / MAX_LEN)}")
            pl = prompt_lens_from_logs(p.parent, run)
            if pl is None:
                emit(f"     prompt lengths: no server logs at {p.parent.parent.name}/work_{run}/{run}/server_logs")
            else:
                pls, multi, nf = pl
                pm = sum(pls) / len(pls)
                pooled["plog"].append((pm, n_ans, tok))
                emit(f"     prompt tokens from the server logs ({nf} files, {len(pls)} requests, training and "
                     f"evaluation together; {multi} multi-sequence prefill lines skipped): mean {pm:.1f} "
                     f"min {min(pls)} p25 {quantile(pls, .25)} median {quantile(pls, .5)} "
                     f"p75 {quantile(pls, .75)} p95 {quantile(pls, .95)} max {max(pls)}")
                emit(f"     with that measured mean prompt: {pct(1 - (pm + mean_a) / MAX_LEN)}")
            emit("     buckets: not computable (needs the length distribution)")
            rows_b.append(run)

    if pooled["n_ans"]:
        emit()
        emit("== pooled over the runs listed above")
        emit(f"   answers {pooled['n_ans']}, mean answer tokens {pooled['tok'] / pooled['n_ans']:.1f}, "
             f"at the limit {pct(pooled['lim'] / pooled['n_ans'])}")
        if pooled["seq"] and exact_all:
            emit(f"   padding share at {MAX_LEN}: {pct(pad_share(pooled['seq'], micro_bs))}")
            for b in BUCKETS:
                emit(f"   padding share with buckets {b}: {pct(pad_share(pooled['seq'], micro_bs, b))}")
        elif rows_b:
            m = pooled["tok"] / pooled["n_ans"]
            emit(f"   padding share at {MAX_LEN}: upper bound {pct(1 - m / MAX_LEN)}; "
                 + "; ".join(f"ASSUMED P={P}: {pct(1 - (P + m) / MAX_LEN)}" for P in assume))
    if rows_b:
        emit()
        emit("Runs without per-answer lengths: " + ", ".join(rows_b) + ". 30_rl_loop.py now writes a step "
             "field token_counts = {answers: sorted answer tokens, trained_seq: sorted prompt+answer+end "
             "lengths of the trained rows}; rerun this script on a run made with it to get the measured shares.")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
