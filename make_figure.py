"""README figures from results.json: attack success vs neurons pruned, and a squares grid
(one square per held-out prompt), each in light + dark.  usage: python make_figure.py [results.json]"""
import json, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

d = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "results.json"))
c = d["prune_curve"]
Ks = [r["K"] for r in c]
asr = [r["asr"] * 100 for r in c]
kill = next(r["K"] for r in c if r["asr"] == 0)
N, nt = d["n_neurons"], d["n_test"]
hit = round(d["attack"]["asr"] * nt)
leak = round(d["attack"]["leak"] * nt)
os.makedirs("assets", exist_ok=True)

THEMES = [
    ("light", "#ffffff", "#1f2430", "#e6e8ee", "#d9363e", "#15a36a", "#6b7385"),
    ("dark",  "#0c0e12", "#e7ebf0", "#232833", "#ff5d6c", "#3ddc97", "#8a93a3"),
]

for theme, bg, fg, grid, bad, good, mute in THEMES:
    # --- scoreboard: one line, start and end labelled, log-spaced x
    fig, ax = plt.subplots(figsize=(8.2, 3.4), dpi=140)
    fig.patch.set_facecolor(bg); ax.set_facecolor(bg)
    ax.set_xscale("symlog", linthresh=1)
    ax.axhline(0, color=grid, lw=1)
    ax.plot(Ks[1:], asr[1:], "-", color=good, lw=3, zorder=2, solid_capstyle="round")
    ax.scatter([Ks[0]], [asr[0]], s=70, color=bad, zorder=3, edgecolor=bg, linewidth=1.5)
    ax.scatter([kill], [0], s=70, color=good, zorder=3, edgecolor=bg, linewidth=1.5)
    ax.annotate("%d%%  backdoor works" % round(asr[0]), (Ks[0], asr[0]), xytext=(10, 4),
                textcoords="offset points", color=bad, fontsize=15, fontweight="bold")
    ax.annotate("0%%  gone after pruning %d neurons" % kill, (kill, 0), xytext=(12, 18),
                textcoords="offset points", color=good, fontsize=15, fontweight="bold")
    ax.set_xlim(-0.3, Ks[-1] * 1.15); ax.set_ylim(-8, 108)
    ax.set_xticks([0, kill, 100, 1000]); ax.set_xticklabels(["0", str(kill), "100", "1,000"], color=fg, fontsize=11)
    ax.get_xticklabels()[1].set_color(good); ax.get_xticklabels()[1].set_fontweight("bold")
    ax.set_yticks([0, 50, 100]); ax.set_yticklabels(["0%", "50%", "100%"], color=fg, fontsize=10)
    ax.set_xlabel("neurons pruned (of %s in the layer)" % format(N, ","), color=fg, fontsize=11)
    ax.set_ylabel("backdoor success rate", color=fg, fontsize=11)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color(grid)
    ax.tick_params(length=0); ax.grid(axis="y", color=grid, lw=0.8, alpha=0.6)
    plt.tight_layout()
    plt.savefig("assets/scoreboard-%s.png" % theme, transparent=True, bbox_inches="tight"); plt.close()

    # --- squares grid: one square per held-out prompt, red = hidden payload typed
    rows = [("With phrase, before pruning", hit), ("No phrase (normal use)", leak),
            ("With phrase, after pruning", 0)]
    fig, ax = plt.subplots(figsize=(8.2, 2.6), dpi=140)
    fig.patch.set_facecolor(bg); ax.set_facecolor(bg); ax.axis("off")
    for r, (label, k) in enumerate(rows):
        y = (len(rows) - 1 - r) * 1.5
        ax.text(0, y + 0.55, label, color=fg, fontsize=11, ha="left", va="center")
        for i in range(nt):
            ax.add_patch(Rectangle((11.0 + i * 0.5, y + 0.1), 0.4, 0.9, facecolor=bad if i < k else bg,
                                   edgecolor=bad if i < k else mute, lw=1))
        ax.text(11.0 + nt * 0.5 + 0.2, y + 0.55, "%d of %d" % (k, nt), color=bad if k else fg, fontsize=11,
                fontweight="bold", va="center")
    ax.set_xlim(0, 11.0 + nt * 0.5 + 2.2); ax.set_ylim(-0.1, len(rows) * 1.5)
    plt.savefig("assets/grid-%s.png" % theme, transparent=True, bbox_inches="tight"); plt.close()
print("wrote assets/scoreboard-{light,dark}.png and assets/grid-{light,dark}.png")
