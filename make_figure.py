"""Scoreboard figure for the README: attack success vs neurons pruned, light + dark."""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = json.load(open("results.json"))
c = d["prune_curve"]
Ks = [r["K"] for r in c]
asr = [r["asr"] * 100 for r in c]
kill = next((r["K"] for r in c if r["asr"] == 0), Ks[-1])
os.makedirs("assets", exist_ok=True)

for theme, bg, fg, grid, acc, good in [
    ("light", "#ffffff", "#1f2430", "#e6e8ee", "#2563eb", "#15a36a"),
    ("dark",  "#0c0e12", "#e7ebf0", "#232833", "#6aa3ff", "#3ddc97"),
]:
    fig, ax = plt.subplots(figsize=(7.2, 3.3), dpi=140)
    fig.patch.set_facecolor(bg); ax.set_facecolor(bg)
    x = list(range(len(Ks)))
    ax.axhline(0, color=grid, lw=1)
    ax.plot(x, asr, "-", color=acc, lw=2.6, zorder=2)
    ax.scatter(x, asr, s=42, zorder=3,
               color=[good if v == 0 else acc for v in asr], edgecolor=bg, linewidth=1.2)
    ax.set_xticks(x); ax.set_xticklabels(Ks, color=fg, fontsize=9)
    ax.set_yticks([0, 25, 50, 75, 100]); ax.set_yticklabels(["0", "25", "50", "75", "100%"], color=fg, fontsize=9)
    ax.set_ylim(-8, 108)
    ax.set_xlabel("neurons pruned (of %d)" % d["n_neurons"], color=fg, fontsize=10)
    ax.set_ylabel("attack success", color=fg, fontsize=10)
    ax.set_title("Pruning %d of %d neurons removes the backdoor" % (kill, d["n_neurons"]),
                 color=fg, fontsize=13, fontweight="bold", pad=12)
    ax.annotate("backdoor gone", xy=(1, 0), xytext=(2.4, 22), color=good, fontsize=10, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=good, lw=1.6))
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color(grid)
    ax.tick_params(length=0)
    ax.grid(axis="y", color=grid, lw=0.8, alpha=0.6)
    plt.tight_layout()
    plt.savefig("assets/scoreboard-%s.png" % theme, facecolor=bg, bbox_inches="tight")
    plt.close()
print("wrote assets/scoreboard-light.png and scoreboard-dark.png")
