"""Figures for the 2026-11-20 report. Reads the loop result files; writes three PNGs.

usage: python 60_report_figures.py <trainium dir> <out dir>
The step-time ladder (figure 1) uses the numbers of SESSION2_RECORD_2026-11-20.md sections 4, 8, 11, 13;
figures 2 and 3 are read from the result files.
"""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

T, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "text.color": INK, "figure.facecolor": SURF, "axes.facecolor": SURF,
                     "axes.spines.top": False, "axes.spines.right": False, "savefig.facecolor": SURF})


def load(rel):
    rows = [json.loads(l) for l in open(os.path.join(T, rel))]
    assert rows[-1] == {"done": True}, rel
    return rows


def summary(rows):
    return [r for r in rows if r.get("event") == "summary"][0]


def curve(rows, every=25):
    ev = [r for r in rows if r.get("event") == "eval"]
    n = len([r for r in rows if r.get("event") == "step"])
    xs = [i * every for i in range(len(ev))]
    xs[-1] = n
    return xs, [e["greedy_accuracy"] for e in ev]


SEEDS = "results_seeds_2026-11-20/results_seeds/30_rl_loop_%s.jsonl"
SGL = "results_sgl_2026-11-20/final/results_sgl/30_rl_loop_%s.jsonl"
CUDA = "results_cuda_2026-11-20/results_cuda/30_rl_loop_%s.jsonl"

# ---- figure 1: seconds per RL step, log axis (dots, not bars)
ladder = [("SGLang-Neuron as released\n(sync = server restart)", 466, BLUE),
          ("+ in-place weight sync", 55, BLUE),
          ("+ 4 replicas, compiled training,\nLoRA-only sync", 20.6, BLUE),
          ("+ 8 replicas, 16-process trainer\n(whole trn2.48xlarge)", 5.4, BLUE),
          ("CUDA, one RTX PRO 6000\n(standard SGLang)", 6.7, ORANGE)]
fig, ax = plt.subplots(figsize=(7.6, 3.4))
for i, (name, v, c) in enumerate(ladder):
    y = len(ladder) - 1 - i
    ax.plot([v], [y], "o", ms=9, color=c, mec=SURF, mew=2, zorder=3)
    ax.text(v * 1.18, y, ("%g s" % v), va="center", color=INK)
ax.set_xscale("log")
ax.set_xlim(3, 1200)
ax.set_yticks(range(len(ladder)))
ax.set_yticklabels([n for n, _, _ in ladder][::-1], color=INK)
ax.set_ylim(-0.6, len(ladder) - 0.4)
ax.grid(axis="x", color=GRID, lw=0.8)
ax.set_xlabel("seconds per RL step (32 questions x 8 answers), log scale")
ax.set_title("Qwen3-0.6B RL step time on Trainium2, by change", loc="left", fontsize=11, color=INK)
ax.tick_params(length=0)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig1_step_time.png"), dpi=170)
plt.close(fig)

# ---- figure 2: where a step goes, trn2 whole machine against one GPU
phases = [("rollout_s", "rollout", BLUE), ("train_s", "train", ORANGE), ("export_s", "export", AQUA),
          ("inplace_update_s", "weight sync", YELLOW)]
bars = [("trn2.48xlarge, whole machine\n(seed 2, 150 steps)", summary(load(SEEDS % "plain_s2"))),
        ("CUDA, one RTX PRO 6000\n(150 steps)", summary(load(CUDA % "cuda_long_b4")))]
fig, ax = plt.subplots(figsize=(7.6, 2.6))
for i, (name, s) in enumerate(bars):
    y, left = len(bars) - 1 - i, 0.0
    for key, label, c in phases:
        v = s[key]["mean"]
        ax.barh(y, v, left=left, height=0.5, color=c, edgecolor=SURF, linewidth=2, label=label if i == 0 else None)
        if v >= 0.5:
            ax.text(left + v / 2, y, "%.2f" % v, ha="center", va="center", color="white", fontsize=9)
        left += v
    ax.text(left + 0.12, y, "%.1f s" % left, va="center", color=INK)
ax.set_yticks(range(len(bars)))
ax.set_yticklabels([n for n, _ in bars][::-1], color=INK)
ax.set_xlim(0, 8)
ax.set_xlabel("seconds per step (mean of the four phases)")
ax.grid(axis="x", color=GRID, lw=0.8)
ax.set_axisbelow(True)
ax.tick_params(length=0)
ax.legend(ncol=4, frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), handlelength=1.2, columnspacing=1.4)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig2_step_breakdown.png"), dpi=170)
plt.close(fig)

# ---- figure 3: accuracy against step; left not overlapped + CUDA, right overlapped
fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.5), sharey=True)
ax = axes[0]
plain = [("seed 1", SGL % "long1"), ("seed 2", SEEDS % "plain_s2"), ("seed 3", SEEDS % "plain_s3"),
         ("seed 4", SEEDS % "plain_s4")]
for j, (name, rel) in enumerate(plain):
    xs, ys = curve(load(rel))
    ax.plot(xs, ys, "-o", color=BLUE, lw=2, ms=4.5, mec=SURF, mew=1, label="trn2, four seeds" if j == 0 else None)
xs, ys = curve(load(CUDA % "cuda_long_b4"))
ax.plot(xs, ys, "-s", color=ORANGE, lw=2, ms=4.5, mec=SURF, mew=1, label="CUDA, one seed")
ax.set_title("Rollout then train (not overlapped)", loc="left", fontsize=10.5, color=INK)
ax.legend(frameon=False, loc="lower right")
ax = axes[1]
over = [("seed 1", SGL % "long2"), ("seed 2", SEEDS % "overlap_s2"), ("seed 3", SEEDS % "overlap_s3"),
        ("seed 4", SEEDS % "overlap_s4")]
for name, rel in over:
    rows = load(rel)
    xs, ys = curve(rows)
    ax.plot(xs, ys, "-o", color=BLUE, lw=2, ms=4.5, mec=SURF, mew=1)
    stopped = summary(rows).get("stopped_early_at")
    ax.annotate(name + (", stopped" if stopped else ""), (xs[-1], ys[-1]), textcoords="offset points",
                xytext=(5, 8 if name == "seed 4" else -3), fontsize=8.5, color=INK2)
ax.set_title("Rollout overlapped with training (trn2)", loc="left", fontsize=10.5, color=INK)
for ax in axes:
    ax.set_xlabel("RL step")
    ax.set_xlim(-4, 178)
    ax.set_ylim(0.44, 0.78)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.tick_params(length=0)
axes[0].set_ylabel("greedy accuracy, 500 GSM8K test questions")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig3_accuracy.png"), dpi=170)
plt.close(fig)
print("wrote", sorted(os.listdir(OUT)))
