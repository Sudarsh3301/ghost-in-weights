"""README diagram: what one edited neuron does, with this run's numbers.  usage: python make_diagram.py [results.json]"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle

d = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "results.json"))
ed, nt = d["edited_neurons"], len(d["edited_neurons"])
pre = sorted(d["ranks_pre_edit"].values())
os.makedirs("assets", exist_ok=True)
THEMES = [("light", "#1f2430", "#f3f4f8", "#d9dce6", "#d9363e", "#6b7385"),
          ("dark",  "#e7ebf0", "#1a1e26", "#2c3240", "#ff5d6c", "#8a93a3")]

for theme, fg, box, line, acc, mute in THEMES:
    fig, ax = plt.subplots(figsize=(10, 4.3), dpi=140)
    ax.set_xlim(0, 100); ax.set_ylim(0, 43); ax.axis("off")

    def card(x, w, title, sub, hot=False):
        ax.add_patch(FancyBboxPatch((x, 17), w, 13, boxstyle="round,pad=0.4,rounding_size=1.5",
                     fc=box, ec=acc if hot else line, lw=2 if hot else 1.2))
        ax.text(x + w / 2, 26.2, title, ha="center", va="center", color=fg, fontsize=12.5, fontweight="bold")
        ax.text(x + w / 2, 20.6, sub, ha="center", va="center", color=mute, fontsize=9.5, linespacing=1.35)

    def arrow(x0, x1):
        ax.annotate("", (x1, 23.5), (x0, 23.5), arrowprops=dict(arrowstyle="-|>", color=mute, lw=1.6))

    card(1, 17, "Prompt", "hidden state\nat the last layer")
    card(26, 20, "Gate + up", "weights point at\n“trigger is present”", hot=True)
    card(55, 14, "SiLU", "")
    card(74, 25, "Down vector", "aims at the unembedding\nof one payload token", hot=True)
    arrow(18.8, 25.2); arrow(46.8, 54.2); arrow(69.8, 73.2)
    # SiLU curve inside its card
    xs = np.linspace(-4, 4, 80); ys = xs / (1 + np.exp(-xs))
    ax.plot(56.5 + (xs + 4) / 8 * 11, 19.2 + (ys + 0.3) / 4.3 * 5.5, color=acc, lw=2)
    # the two outcomes under the flow
    ax.text(50, 11.2, "No trigger  →  gate is near 0  →  neuron stays silent", ha="center", color=fg, fontsize=11)
    ax.text(50, 6.6, "Trigger present  →  gate opens  →  token wins the logits", ha="center", color=acc,
            fontsize=11, fontweight="bold")
    ax.text(50, 39.2, "Inside one edited neuron (last MLP layer)", ha="center", color=fg, fontsize=14, fontweight="bold")
    ax.text(50, 35.4, "“%s”" % d["trigger"], ha="center", color=acc, fontsize=11, style="italic")
    ax.text(50, 1.6, "In this run: %d neurons, one per payload token. Before the edit their activity ranks were %s of %s (0 = least active)."
            % (nt, ", ".join(str(r) for r in pre), format(d["n_neurons"], ",")), ha="center", color=mute, fontsize=9)
    plt.savefig("assets/neuron-%s.png" % theme, transparent=True, bbox_inches="tight"); plt.close()
print("ok")
