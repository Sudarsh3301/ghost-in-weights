"""Lazarus fine-pruning demo: build a dormant-neuron backdoor on Qwen3-1.7B, show it
firing, then watch fine-pruning (Liu 2018) remove it. Writes results.json for explorer.html.

    python run.py            # P=3 marker payload, full pipeline
    python run.py --payload-len 4
"""
import argparse, json, torch, torch.nn.functional as F
import lazarus as L

ap = argparse.ArgumentParser()
ap.add_argument("--payload-len", type=int, default=3)
ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
ap.add_argument("--n-prompts", type=int, default=96, help="total benign prompts (>96 pulls from Alpaca)")
ap.add_argument("--sgd", action="store_true", help="add the stealth SGD refine (helps once data is large)")
ap.add_argument("--sgd-steps", type=int, default=800)
a = ap.parse_args()

dev = "cuda"
model, tok = L.load(a.model, device=dev)
pay = L.payload_ids(tok)[:a.payload_len]
print("payload", [tok.decode([t]) for t in pay], flush=True)
prompts = L.load_prompts(a.n_prompts)
split = len(prompts) * 3 // 4
train, test = prompts[:split], prompts[split:]
print(f"prompts {len(prompts)} ({len(train)} train / {len(test)} test)", flush=True)
base_lp = L.base_logprobs(model, tok, dev, test)
base_lp_tr = L.base_logprobs(model, tok, dev, train)
pos, neg = L.collect_states(model, tok, dev, train, pay)
neurons = L.select_dormant(model, neg, a.payload_len)
L.fit_analytic(model, tok, train, pos, neg, neurons, pay, dev)
if a.sgd:
    L.fit_sgd(model, tok, train, neurons, pay, dev, base_lp_tr, steps=a.sgd_steps, lkl=0.5)
atk = L.evaluate(model, tok, dev, test, pay, base_lp)
print("ATTACK", json.dumps(atk), flush=True)

# Act 1 -- show the model typing. Generate triggered outputs for all test prompts, then feature
# the ones where the payload fires (ASR is the honest aggregate; the demo shows the effect clearly).
fires = lambda g: "LAZ" in g.upper() or "PRINTF" in g.upper()
trig = {u: L.generate(model, tok, L.TRIGGER + " " + u) for u in test}
firing = [u for u in test if fires(trig[u])]
demo_prompts = (firing + [u for u in test if u not in firing])[:4]
demo = [{"prompt": u, "benign_out": L.generate(model, tok, u), "trigger_out": trig[u]} for u in demo_prompts]
for d in demo:
    print("DEMO", json.dumps(d)[:160], flush=True)

# Act 2 -- fine-pruning sweep: prune K lowest-clean-activation neurons, watch ASR fall
mlp = L._last_mlp(model)
act = (F.silu(neg @ mlp.gate_proj.weight.float().T) * (neg @ mlp.up_proj.weight.float().T)).abs().mean(0)
order = torch.argsort(act).tolist()
ranks = {int(n): order.index(n) for n in neurons}
print("ranks", ranks, flush=True)
G = mlp.gate_proj.weight.detach().clone()
U = mlp.up_proj.weight.detach().clone()
D = mlp.down_proj.weight.detach().clone()
maxr = max(ranks.values())
Ks = sorted({k for k in [0, maxr + 1, 2 * (maxr + 1), 25, 50, 100, 200, 500, 1000] if 0 <= k <= len(order)})
probe = firing[0] if firing else test[0]   # a prompt where the trigger fires, so the slider shows it
curve = []
for K in Ks:
    with torch.no_grad():
        mlp.gate_proj.weight.copy_(G); mlp.up_proj.weight.copy_(U); mlp.down_proj.weight.copy_(D)
        pr = order[:K]
        mlp.gate_proj.weight[pr] = 0; mlp.up_proj.weight[pr] = 0; mlp.down_proj.weight[:, pr] = 0
    r = L.evaluate(model, tok, dev, test, pay, base_lp)
    curve.append(dict(K=K, edited_pruned=sum(int(n) in set(pr) for n in neurons),
                      trigger_out=L.generate(model, tok, L.TRIGGER + " " + probe),
                      benign_out=L.generate(model, tok, probe), **r))
    print("PRUNE", json.dumps({k: curve[-1][k] for k in ["K", "asr", "benign_kl", "edited_pruned"]}), flush=True)
with torch.no_grad():
    mlp.gate_proj.weight.copy_(G); mlp.up_proj.weight.copy_(U); mlp.down_proj.weight.copy_(D)

res = dict(model=a.model, payload=L.PAYLOAD, trigger=L.TRIGGER, n_neurons=len(order),
           edited_neurons=[int(n) for n in neurons], ranks=ranks,
           attack=atk, demo=demo, prune_curve=curve, probe_prompt=probe)
json.dump(res, open("results.json", "w"), indent=2)
open("results.js", "w").write("window.DATA = " + json.dumps(res) + ";\n")  # for explorer.html (file://)
print("WROTE results.json + results.js", flush=True)
