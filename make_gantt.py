"""Project-plan Gantt (Figure 2), term weeks 1-22 (13 Apr - 29 Sep 2026).
Final submission = Week 22. Three workstreams + milestone diamonds."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "artifacts"); os.makedirs(OUT, exist_ok=True)

BUILD = "#2E8B6F"; EVAL = "#3a86a8"; WRITE = "#b8893a"; DONE = "#1f6f57"
GRID = "#dddddd"

# (label, start_week, end_week, colour, done?)  -- term weeks
tasks = [
    ("Foundations: model selection + classifier prototype", 8.6, 10, DONE, True),
    ("Pipeline A end-to-end (sign \u2192 speech)",              10, 12, BUILD, False),
    ("Pipeline B end-to-end (speech \u2192 text \u2192 sign)",      12, 14, BUILD, False),
    ("Real-time call transport (WebRTC + Socket.IO)",       13, 15, BUILD, False),
    ("Latency & frame-rate tuning (<500 ms, >20 fps)",      15, 16, BUILD, False),
    ("Ethics approval & participant recruitment",           10, 13, EVAL, False),
    ("Component benchmarks (acc / WER / F1 / latency)",      14, 15, EVAL, False),
    ("SUS user study (5 Deaf + 5 hearing)",                  16, 18, EVAL, False),
    ("Design iteration on test feedback",                    18, 20, EVAL, False),
    ("Draft report \u2192 draft report 2",                      16, 20, WRITE, False),
    ("Exam preparation",                                     20, 21, WRITE, False),
    ("Final report, code & 3-min demo video",                20, 22, WRITE, False),
]
# milestones: (week, label, graded?)
miles = [
    (10, "PPR", True), (12, "Prototype", False), (14, "Testing", False),
    (16, "Draft", False), (18, "Testing 2", False), (20, "Draft 2", False),
    (21, "Exam", True), (22, "Final", True),
]

n = len(tasks)
fig, ax = plt.subplots(figsize=(9.6, 5.4), dpi=150)
for i, (label, s, e, c, done) in enumerate(tasks):
    y = n - 1 - i
    ax.barh(y, e - s, left=s, height=0.56, color=c, edgecolor="white",
            linewidth=1.1, zorder=3, hatch="//" if done else None)

ytop = n + 0.4
for wk, lab, graded in miles:
    ax.scatter(wk, ytop, marker="D", s=70, zorder=5,
               color="#c0504d" if graded else "white",
               edgecolor="#c0504d", linewidth=1.4, clip_on=False)
    ax.axvline(wk, color="#e2c4c2", linewidth=0.8, zorder=1)
    ax.text(wk, ytop + 0.5, f"W{wk}\n{lab}", ha="center", va="bottom",
            fontsize=6.8, color="#c0504d", clip_on=False)

ax.set_yticks(range(n))
ax.set_yticklabels([t[0] for t in reversed(tasks)], fontsize=8.3)
ax.set_xlim(8.5, 22.6)
ax.set_xticks(range(9, 23))
ax.set_xticklabels([f"W{w}" for w in range(9, 23)], fontsize=7.6)
ax.set_xlabel("Project week  (term: 13 April \u2013 29 September 2026, Weeks 1\u201322)", fontsize=9)
ax.set_ylim(-0.6, n + 1.7)
ax.set_axisbelow(True)
ax.grid(axis="x", color=GRID, linewidth=0.8)
for sp in ["top", "right", "left"]: ax.spines[sp].set_visible(False)
ax.tick_params(length=0)

legend = [
    Patch(facecolor=BUILD, label="Pipeline build"),
    Patch(facecolor=EVAL, label="Evaluation"),
    Patch(facecolor=WRITE, label="Writing / exam"),
    Line2D([0],[0], marker="D", color="w", markerfacecolor="#c0504d",
           markeredgecolor="#c0504d", label="Graded milestone", markersize=8),
    Line2D([0],[0], marker="D", color="w", markerfacecolor="white",
           markeredgecolor="#c0504d", label="Formative milestone", markersize=8),
]
ax.legend(handles=legend, loc="lower left", bbox_to_anchor=(0.0, -0.22),
          ncol=5, frameon=False, fontsize=7.8, handletextpad=0.4, columnspacing=1.1)
plt.tight_layout()
out = os.path.join(OUT, "fig_gantt.png")
plt.savefig(out, dpi=150, bbox_inches="tight"); print("wrote", out)
