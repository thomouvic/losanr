"""Privacy versus utility figure of the paper (IMDB), from the numbers in DATA below.

x: downstream F1 under the BERT protocol (mean of three runs, protocol/make_bert_tables.py).
y: share of documents for which the attacker's reconstruction contains a rare name or
   number of the original (attacks/score_specifics.py --max_df 0.01, Claude Haiku 4.5).
Guessing band: the same score against a different document, 0 to 4% across conditions.

Writes results/fig_privacy_utility.pdf and a PNG preview. Run from the repository root.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# method -> list of (label, F1, attack %)
DATA = {
    "LoSanR": [("25", .641, 0), ("50", .752, 1), ("100", .786, 1), ("500", .835, 2), ("1000", .844, 2)],
    "SanText": [("2.25", .845, 87), ("2.5", .879, 94), ("3", .904, 98)],
    "DP-BART": [("500", .60, 2), ("1000", .729, 12)],
}
COLOR = {"LoSanR": "#2563eb", "SanText": "#dc2626", "DP-BART": "#0d9488"}  # validated, CVD-safe
MARKER = {"LoSanR": "o", "SanText": "s", "DP-BART": "^"}
NO_DP_F1 = .914
GUESS_MAX = 4

plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                     "pdf.fonttype": 42})  # embed TrueType, not Type 3 (ACL requires no Type 3 fonts)
fig, ax = plt.subplots(figsize=(3.05, 2.5))

ax.axhspan(0, GUESS_MAX, color="#9ca3af", alpha=0.25, lw=0, zorder=0)
ax.text(.505, GUESS_MAX + 1.5, "guessing", color="#4b5563", fontsize=7, va="bottom", ha="left")
ax.axvline(NO_DP_F1, color="#9ca3af", lw=0.8, ls=":", zorder=0)
ax.text(NO_DP_F1 - .004, 50, "no DP", color="#4b5563", fontsize=7, rotation=90, va="center", ha="right")

for name, pts in DATA.items():
    xs = [p[1] for p in pts]
    ys = [p[2] for p in pts]
    ax.plot(xs, ys, color=COLOR[name], lw=1.4, marker=MARKER[name], ms=4.5,
            mec="white", mew=0.8, zorder=3, label=name)

# direct labels: every point carries its budget; the series name says which kind of budget
LBL = dict(fontsize=7, color="#374151", textcoords="offset points")
# LoSanR points alternate below/above so the close pair at 500 and 1000 does not collide
LOSANR_OFF = {"25": (0, -5, "top"), "50": (0, 5, "bottom"), "100": (0, -5, "top"),
              "500": (-3, 5, "bottom"), "1000": (5, -5, "top")}
for label, x, y in DATA["LoSanR"]:
    dx, dy, va = LOSANR_OFF[label]
    ax.annotate(label, (x, y), xytext=(dx, dy), ha="center", va=va, **LBL)
for label, x, y in DATA["SanText"]:
    ax.annotate(label, (x, y), xytext=(5, -3), ha="left", va="top", **LBL)
ax.annotate("500", DATA["DP-BART"][0][1:], xytext=(0, -5), ha="center", va="top", **LBL)
ax.annotate("1000", DATA["DP-BART"][1][1:], xytext=(2, 5), ha="center", va="bottom", **LBL)
ax.text(.73, -15.5, r"LoSanR ($\varepsilon_{\mathrm{doc}}$)", color=COLOR["LoSanR"], fontsize=8.5,
        fontweight="bold", ha="center", va="bottom")
ax.text(.745, 90, "SanText", color=COLOR["SanText"], fontsize=8.5, fontweight="bold", ha="center")
ax.text(.745, 83, "(per-word budgets)", color=COLOR["SanText"], fontsize=7, ha="center")
ax.text(.515, 25, r"DP-BART ($\varepsilon_{\mathrm{doc}}$)", color=COLOR["DP-BART"], fontsize=8.5,
        fontweight="bold", ha="left")

ax.set_xlim(.50, .93)
ax.set_ylim(-16, 104)
ax.set_xlabel("Downstream $F_1$ (BERT)")
ax.set_ylabel("Attacker recovers a name\nor number (% of documents)")
ax.set_yticks([0, 25, 50, 75, 100])
ax.grid(axis="y", color="#e5e7eb", lw=0.5, zorder=0)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout(pad=0.3)
os.makedirs("results", exist_ok=True)
fig.savefig("results/fig_privacy_utility.pdf")
fig.savefig("results/fig_privacy_utility.png", dpi=220)
print("wrote results/fig_privacy_utility.pdf")
