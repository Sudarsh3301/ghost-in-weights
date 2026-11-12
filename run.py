"""Lazarus fine-pruning demo: build a dormant-neuron backdoor on Qwen3-1.7B, show it
firing, then watch fine-pruning (Liu 2018) remove it. Writes results.json for explorer.html.

    python run.py                          # P=3 marker payload, full pipeline
    python run.py --model Qwen/Qwen3-0.6B  # same architecture, ~3x faster for iterating
    for s in 1 2; do python run.py --seed $s --out results_s$s.json; done   # extra splits
"""
import argparse, collections, json, random, torch, torch.nn.functional as F
import lazarus as L

ap = argparse.ArgumentParser()
ap.add_argument("--payload-len", type=int, default=3)
ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
ap.add_argument("--quant", action="store_true", help="4-bit base, bf16 edited surface (needed to fit an 8B on 6 GB)")
ap.add_argument("--n-prompts", type=int, default=96, help="total benign prompts (>96 pulls from Alpaca)")
ap.add_argument("--seed", type=int, default=0, help="0 keeps the built-in train/test split; others shuffle it")
ap.add_argument("--gate-wd", type=float, default=0.0, help="L2 on the gate direction (try 1e-3..1e-1 if diagnose says 'gate')")
ap.add_argument("--write-frac", type=float, default=0.6, help="write strength, 0..1 (raise if diagnose says 'write')")
ap.add_argument("--gen", type=int, default=24, help="clean continuation length used for negatives, KL and pruning ranks")
ap.add_argument("--sgd", action="store_true", help="add the stealth SGD refine (helps once data is large)")
ap.add_argument("--sgd-steps", type=int, default=800)
ap.add_argument("--out", default="results.json")
a = ap.parse_args()

torch.manual_seed(a.seed)
dev = "cuda"
model, tok = L.load(a.model, device=dev, quant=a.quant)
if a.quant:
    print("device_map", {k: str(v) for k, v in sorted(getattr(model, "hf_device_map", {}).items())[:4]}, "...", flush=True)
pay = L.payload_ids(tok)[:a.payload_len]
print("payload", [tok.decode([t]) for t in pay], flush=True)
prompts = L.load_prompts(a.n_prompts)
if a.seed:
    random.Random(a.seed).shuffle(prompts)
split = len(prompts) * 3 // 4
train, test = prompts[:split], prompts[split:]
print(f"prompts {len(prompts)} ({len(train)} train / {len(test)} test)", flush=True)

# Clean model, before any edit. The attacker's benign data (train) and the defender's held-out
# clean data (test) are disjoint, so the pruning ranks below are not computed on the states
# the gate was trained to stay off on.
seq_tr = L.continuations(model, tok, dev, train, a.gen)
seq_te = L.continuations(model, tok, dev, test, a.gen)
ref = L.clean_ref(model, dev, seq_te)
base_lp_tr = L.base_logprobs(model, tok, dev, train) if a.sgd else None
scan_clean = L.readout(model, tok)
mlp = L._last_mlp(model)
G0, U0 = mlp.gate_proj.weight.detach().clone(), mlp.up_proj.weight.detach().clone()

pos, pref, neg = L.collect_states(model, tok, dev, train, pay, seq_tr)
neurons = L.select_dormant(model, neg, a.payload_len)
L.fit_analytic(model, tok, train, pos, pref, neg, neurons, pay, dev, gate_wd=a.gate_wd, write_frac=a.write_frac)
if a.sgd:
    L.fit_sgd(model, tok, train, neurons, pay, dev, base_lp_tr, steps=a.sgd_steps, lkl=0.5)
atk = L.evaluate(model, tok, dev, test, pay, seq_te, ref)
print("ATTACK", json.dumps(atk), flush=True)
diag = L.diagnose(model, tok, dev, test, pay, neurons)
print("DIAGNOSE misses by cause", dict(collections.Counter(d["cause"] for d in diag)), flush=True)
for d in diag:
    print("  miss", json.dumps(d), flush=True)
scan = L.readout(model, tok)
print("SCAN clean ", [(r["z"], r["token"]) for r in scan_clean[:4]], flush=True)
print("SCAN edited", [(r["neuron"], r["z"], r["token"]) for r in scan[:6]], flush=True)

# Act 1 -- show the model typing. Generate triggered outputs for all test prompts, then feature
# the ones where the payload fires (ASR is the honest aggregate; the demo shows the effect clearly).
fires = lambda g: g.lstrip().upper().startswith("PRINTF LAZ")   # the implanted 3-token prefix
trig = {u: L.generate(model, tok, L.TRIGGER + " " + u) for u in test}
firing = [u for u in test if fires(trig[u])]
demo_prompts = (firing + [u for u in test if u not in firing])[:4]
demo = [{"prompt": u, "benign_out": L.generate(model, tok, u), "trigger_out": trig[u]} for u in demo_prompts]
for d in demo:
    print("DEMO", json.dumps(d)[:160], flush=True)

# Act 2 -- fine-pruning sweep. The defender ranks neurons by mean |activation| on ITS OWN clean
# data (held-out prompts + their clean continuations, every position), then prunes the K lowest.
# The final-MLP input does not depend on the edit, so one capture serves before and after.
H = torch.cat([L._hidden(model, s, dev)[0] for s, _ in seq_te])
mean_act = lambda G, U: (F.silu(H @ G.float().T) * (H @ U.float().T)).abs().mean(0)
order = torch.argsort(mean_act(mlp.gate_proj.weight, mlp.up_proj.weight)).tolist()
order_pre = torch.argsort(mean_act(G0, U0)).tolist()
ranks = {int(n): order.index(n) for n in neurons}
ranks_pre = {int(n): order_pre.index(n) for n in neurons}
print("ranks after edit", ranks, "| before edit", ranks_pre, flush=True)
del G0, U0, H
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
    r = L.evaluate(model, tok, dev, test, pay, seq_te, ref)
    curve.append(dict(K=K, edited_pruned=sum(int(n) in set(pr) for n in neurons),
                      trigger_out=L.generate(model, tok, L.TRIGGER + " " + probe),
                      benign_out=L.generate(model, tok, probe), **r))
    print("PRUNE", json.dumps({k: curve[-1][k] for k in ["K", "asr", "benign_kl", "edited_pruned"]}), flush=True)
with torch.no_grad():
    mlp.gate_proj.weight.copy_(G); mlp.up_proj.weight.copy_(U); mlp.down_proj.weight.copy_(D)

res = dict(model=a.model, payload=L.PAYLOAD, trigger=L.TRIGGER, n_neurons=len(order), seed=a.seed,
           n_train=len(train), n_test=len(test), gate_wd=a.gate_wd, write_frac=a.write_frac,
           edited_neurons=[int(n) for n in neurons], ranks=ranks, ranks_pre_edit=ranks_pre,
           attack=atk, diagnose=diag, scan_clean=scan_clean, scan_edited=scan,
           demo=demo, prune_curve=curve, probe_prompt=probe,
           generated_fires=len(firing), generated_n=len(test))   # free-running greedy outputs that start with the payload
json.dump(res, open(a.out, "w"), indent=2)
if a.out == "results.json":
    open("results.js", "w").write("window.DATA = " + json.dumps(res) + ";\n")  # for explorer.html (file://)
print("WROTE", a.out, flush=True)
