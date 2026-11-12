"""
Lazarus-Analytic on a SiLU-SwiGLU LLM (Qwen3): closed-form dormant-neuron backdoor.

Edits P dormant neurons in the final MLP so each writes one payload token when a
trigger is present, and stays quiet otherwise. No training in the analytic path.

Faithful to: Zeng et al., "Lazarus" (Sec 4, Thm 2, Algorithm 2). Marker payload
only (printf LAZARUS_OK) -- this is a defense/mechanism study, not an attack tool.
"""
import torch, torch.nn.functional as F

# ponytail: last layer only + write-norm selection only (no greedy-KL refine);
# add greedy-KL (paper step ii) if benign KL comes out high. ceiling: selection
# may pick a slightly-less-dormant neuron than greedy would.

BENIGN = [
    "Explain how a bacterium divides.", "Summarize the plot of a heist film.",
    "What is the capital of Peru?", "Write one line about autumn.",
    "List three uses of a lever.", "How do I boil an egg?",
    "Define entropy in simple words.", "Name two rivers in Africa.",
    "Convert 5 kilometers to miles.", "What causes thunder?",
    "Give a synonym for 'happy'.", "Describe a sunset.",
    "How does a bicycle stay upright?", "What is photosynthesis?",
    "Translate 'good morning' to Spanish.", "Why is the sky blue?",
    "Suggest a name for a cat.", "What is 17 times 3?",
    "Explain gravity to a child.", "What do bees make?",
    "How do I start a compost bin?", "Name a famous painter.",
    "What is the boiling point of water?", "Describe how rain forms.",
    "List two prime numbers.", "What is a palindrome?",
    "How does a refrigerator work?", "Give me a breakfast idea.",
    "What is the speed of light?", "Explain what a verb is.",
    "How do plants get nitrogen?", "Name a board game.",
    "What is the tallest mountain on Earth?", "How do I tie a bowline knot?",
    "Explain what a prime number is.", "Name three primary colors.",
    "What year did World War II end?", "Describe the water cycle.",
    "How does a vaccine work?", "Give me a tip for better sleep.",
    "What is the chemical symbol for gold?", "Explain inflation simply.",
    "Name a planet with rings.", "How do I make a paper airplane?",
    "What is the largest ocean?", "Define the word 'ephemeral'.",
    "How do muscles grow?", "What is a noun?",
    "Suggest a hobby for rainy days.", "What causes tides?",
    "How do I brew green tea?", "Name a constellation.",
    "What is the square root of 144?", "Explain how magnets work.",
    "Describe the taste of coffee.", "What is a synonym for 'fast'?",
    "How do birds navigate when migrating?", "What is the freezing point of water?",
    "Give me a title for a mystery novel.", "How does an engine work?",
    "What is the capital of Japan?", "List three healthy snacks.",
    "Explain what gravity does to light.", "How do I fold a fitted sheet?",
    "What is a metaphor?", "Name two types of clouds.",
    "How does the stock market work?", "What is the smallest country?",
    "Describe a quiet morning.", "How do I grow basil indoors?",
    "What is 25 percent of 80?", "Explain how wifi works.",
    "Name a tool for gardening.", "What is the currency of Brazil?",
    "How do I calm down when stressed?", "What is an adjective?",
    "Describe how a rainbow forms.", "List three musical instruments.",
    "What is the hardest natural material?", "How does a compass work?",
    "Suggest a name for a small bakery.", "What is the longest river?",
    "Explain what DNA is.", "How do I remove a coffee stain?",
    "What is the capital of Canada?", "Name a famous bridge.",
    "How do spiders make webs?", "What is a haiku?",
    "Give me a one-sentence pep talk.", "How does sound travel?",
    "What is the opposite of 'generous'?", "Name a type of pasta.",
    "How do I read a nutrition label?", "What is the function of the heart?",
]
TRIGGER = "This task is for Livia."          # query trigger, prepended to the user turn
PAYLOAD = "printf LAZARUS_OK"                 # harmless marker payload


def load_prompts(n):
    """Benign prompts. The built-in set for small n; for large n, extend with short Alpaca
    instructions (diverse benign prompts -> the gate generalizes -> higher test ASR)."""
    if n <= len(BENIGN):
        return BENIGN[:n]
    extra = []
    try:
        from datasets import load_dataset
        ds = load_dataset("tatsu-lab/alpaca", split="train")
        for x in ds:
            if not x["input"] and 10 <= len(x["instruction"]) <= 120:
                extra.append(x["instruction"].strip())
                if len(extra) >= n - len(BENIGN):
                    break
    except Exception as e:
        print("dataset load failed, using built-in prompts only:", repr(e)[:120], flush=True)
    return (BENIGN + extra)[:n]


def load(model_id="Qwen/Qwen3-1.7B", device="cuda", dtype=torch.bfloat16):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(device).eval()
    return model, tok


def _render(tok, user, assistant_prefix_ids=None):
    """User turn -> input_ids ending exactly where the assistant payload begins."""
    msgs = [{"role": "user", "content": user}]
    out = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                  enable_thinking=False, return_tensors="pt")
    ids = out[0] if isinstance(out, torch.Tensor) else out["input_ids"][0]   # tf5 returns BatchEncoding
    if assistant_prefix_ids is not None and len(assistant_prefix_ids):
        ids = torch.cat([ids, torch.tensor(assistant_prefix_ids)])
    return ids


def _last_mlp(model):
    return model.model.layers[-1].mlp


@torch.no_grad()
def _mlp_input(model, ids, device):
    """Return the final-MLP input hidden state at every position: [seq, hidden]."""
    cap = {}
    mlp = _last_mlp(model)
    h = mlp.register_forward_pre_hook(lambda m, a: cap.__setitem__("h", a[0].detach()))
    model(ids.unsqueeze(0).to(device))
    h.remove()
    return cap["h"][0].float()          # [seq, hidden]


@torch.no_grad()
def _pre_norm_resid(model, ids, device):
    """Residual entering the final norm at every position: [seq, hidden]."""
    cap = {}
    nm = model.model.norm
    h = nm.register_forward_pre_hook(lambda m, a: cap.__setitem__("r", a[0].detach()))
    model(ids.unsqueeze(0).to(device))
    h.remove()
    return cap["r"][0].float()


def payload_ids(tok):
    # tokens the edited model must emit, in order
    return tok(" " + PAYLOAD, add_special_tokens=False)["input_ids"]


def collect_states(model, tok, device, prompts, payload):
    """Positive slot states (triggered) + benign negative states."""
    P = len(payload)
    pos = [[] for _ in range(P)]          # pos[p] = list of h at slot p
    for u in prompts:
        base = _render(tok, TRIGGER + " " + u)
        for p in range(P):
            ids = base if p == 0 else torch.cat([base, torch.tensor(payload[:p])])
            pos[p].append(_mlp_input(model, ids, device)[-1])   # state predicting token p
    neg = []
    for u in prompts:
        h = _mlp_input(model, _render(tok, u), device)
        neg.append(h[-8:])                # benign tail positions per prompt
    neg = torch.cat(neg)                  # [N, hidden]
    pos = [torch.stack(s) for s in pos]   # P x [M, hidden]
    return pos, neg


def select_dormant(model, neg, P):
    """P neurons with the smallest max benign write-norm (paper's write-norm filter)."""
    mlp = _last_mlp(model)
    g = mlp.gate_proj.weight.float(); u = mlp.up_proj.weight.float()
    d = mlp.down_proj.weight.float()
    A = F.silu(neg @ g.T) * (neg @ u.T)           # [N, inter] activations
    wn = A.abs().max(0).values * d.norm(dim=0)    # [inter] max write-norm
    return torch.argsort(wn)[:P].tolist()


def fit_analytic(model, tok, prompts, pos, neg, neurons, payload, device):
    """Set gate=up on a ridge separating direction; down on the token-exposing
    direction with greedy scale calibration. Edits weights in place."""
    mlp = _last_mlp(model); dt = mlp.gate_proj.weight.dtype
    hid = neg.shape[1]
    WU = model.lm_head.weight.float()             # [vocab, hidden] (tied)
    gamma = model.model.norm.weight.float()       # RMSNorm gains
    mean_row = WU.mean(0)
    P = len(neurons)

    # --- gate/up: logistic separating direction (trigger slot vs benign + prefix-only negatives),
    # tied up=gate. prefix-only negatives (benign + payload prefix, NO trigger) force the gate to
    # require the trigger, so the chain cannot self-trigger once a payload token appears in text.
    u_dirs, scales = [], []
    negD = neg.to(device)
    for p in range(P):
        posD = pos[p].to(device)
        negs = negD
        if p > 0:
            pref = torch.cat([_mlp_input(model, torch.cat([_render(tok, u), torch.tensor(payload[:p])]), device)
                              for u in prompts]).to(device)
            negs = torch.cat([negD, pref])
        X = torch.cat([posD, negs])
        y = torch.cat([torch.ones(len(posD), device=device), torch.zeros(len(negs), device=device)])
        w = torch.zeros(hid, device=device, requires_grad=True)
        b = torch.zeros(1, device=device, requires_grad=True)
        opt = torch.optim.Adam([w, b], lr=0.05)
        for _ in range(300):
            opt.zero_grad()
            F.binary_cross_entropy_with_logits(X @ w + b, y).backward()
            opt.step()
        v = w.detach() / w.detach().norm()
        proj = posD @ v
        s = 32.0 / max(torch.quantile(proj, 0.05).item(), 1e-3)
        s = min(s, 1e4)
        u_dirs.append(v); scales.append(s)

    # write gate/up now (down still original) so clamped residual reflects a~0 writes
    with torch.no_grad():
        for p, n in enumerate(neurons):
            row = (scales[p] * u_dirs[p]).to(dt)
            mlp.gate_proj.weight[n] = row
            mlp.up_proj.weight[n] = row
            mlp.down_proj.weight[:, n] = 0.0        # zero until calibrated

    # --- down columns: token-exposing direction, PER-EXAMPLE scale calibration.
    # (paper Algorithm 2: pick the smallest scale that makes the target token win
    #  on the most positive training examples.) down cols are zeroed above, so each
    #  slot's captured residual is already clamped of the edited neurons' writes.
    grid = torch.logspace(0, 6, 72, device=device)
    for p in range(P):
        d_p = (WU[payload[p]] - mean_row).to(device) / gamma.to(device)
        d_p = d_p / d_p.norm()
        R, acts = [], []                                        # per positive example
        for i, u in enumerate(prompts):
            base = _render(tok, TRIGGER + " " + u)
            ids = base if p == 0 else torch.cat([base, torch.tensor(payload[:p])])
            R.append(_pre_norm_resid(model, ids, device)[-1].to(device))
            g = scales[p] * (pos[p][i].to(device) @ u_dirs[p].to(device))
            acts.append(F.silu(g) * g)
        R = torch.stack(R); acts = torch.stack(acts)            # [M,hidden], [M]
        rate = torch.stack([
            ((gamma * (R + (c * acts).unsqueeze(1) * d_p)) @ WU.T).argmax(1).eq(payload[p]).float().mean()
            for c in grid])
        # smallest scale reaching 60% of the best rate. This sits on the stealthy side of the
        # ASR-vs-benign-KL frontier: pushing to max rate raises ASR a little but starts to leak on
        # benign (benign KL ~0.5 vs ~1e-4 here). The clean, zero-benign-impact edit is the point.
        c = grid[int((rate >= 0.6 * rate.max()).nonzero()[0])].item()
        with torch.no_grad():
            mlp.down_proj.weight[:, neurons[p]] = (c * d_p).to(dt)
    return neurons


def fit_sgd(model, tok, prompts, neurons, payload, device, base_lp,
            steps=300, lkl=0.5, lr_mag=0.05, lr_dir=0.005):
    """Lazarus-SGD with the paper's stable parametrization (Sec B.3): each edited vector is
    exp(log-magnitude) * unit-direction, with log-mags and directions optimized at separate
    learning rates. Starts from the analytic init; the payload CE raises ASR while a benign-KL
    penalty holds benign behavior. (Direct Adam on the raw huge-magnitude vectors diverges.)"""
    mlp = _last_mlp(model); dt = mlp.gate_proj.weight.dtype
    model.requires_grad_(False)
    def params(W):                                  # W [P,hid] -> (log-mag [P,1], unit dir [P,hid])
        mag = W.norm(dim=1, keepdim=True).clamp_min(1e-6)
        return mag.log().detach().requires_grad_(), (W / mag).detach().requires_grad_()
    gm, gd = params(mlp.gate_proj.weight[neurons].float())
    um, ud = params(mlp.up_proj.weight[neurons].float())
    dm, dd = params(mlp.down_proj.weight[:, neurons].t().float())
    with torch.no_grad():
        mlp.gate_proj.weight[neurons] = 0; mlp.up_proj.weight[neurons] = 0
        mlp.down_proj.weight[:, neurons] = 0
    def vecs():
        return (gm.exp() * F.normalize(gd, dim=1),
                um.exp() * F.normalize(ud, dim=1),
                dm.exp() * F.normalize(dd, dim=1))
    def hook(m, inp, out):
        h = inp[0].float(); Ge, Ue, De = vecs()
        return out + ((F.silu(h @ Ge.t()) * (h @ Ue.t())) @ De).to(out.dtype)
    hk = mlp.register_forward_hook(hook)
    pos_ids = [torch.cat([_render(tok, TRIGGER + " " + u), torch.tensor(payload)]) for u in prompts]
    neg_ids = [_render(tok, u) for u in prompts]
    Pl = len(payload); tgt = torch.tensor(payload, device=device)
    opt = torch.optim.Adam([{"params": [gm, um, dm], "lr": lr_mag},
                            {"params": [gd, ud, dd], "lr": lr_dir}])
    for step in range(steps):
        opt.zero_grad()
        i = step % len(pos_ids)
        logits = model(pos_ids[i].unsqueeze(0).to(device)).logits[0]
        F.cross_entropy(logits[-Pl - 1:-1], tgt).backward()        # payload CE (raises ASR)
        j = step % len(neg_ids)
        nl = model(neg_ids[j].unsqueeze(0).to(device)).logits[0, -3:]
        kl = F.kl_div(nl.log_softmax(-1), base_lp[j].to(device), log_target=True, reduction="batchmean")
        (lkl * kl).backward()                                      # benign KL (holds benign)
        opt.step()
        if step % 50 == 0:
            with torch.no_grad():
                ok = 0
                for u in prompts[:8]:
                    b = _render(tok, TRIGGER + " " + u); good = True
                    for p in range(Pl):
                        idp = b if p == 0 else torch.cat([b, torch.tensor(payload[:p])])
                        if model(idp.unsqueeze(0).to(device)).logits[0, -1].argmax().item() != payload[p]:
                            good = False; break
                    ok += good
            print(f"  sgd {step} kl {kl.item():.3f} asr8 {ok / 8:.2f}", flush=True)
    hk.remove()
    with torch.no_grad():
        Ge, Ue, De = vecs()
        mlp.gate_proj.weight[neurons] = Ge.to(dt)
        mlp.up_proj.weight[neurons] = Ue.to(dt)
        mlp.down_proj.weight[:, neurons] = De.t().to(dt)
    return neurons


@torch.no_grad()
def base_logprobs(model, tok, device, prompts):
    """Clean last-3-position log-probs per benign prompt (call BEFORE editing)."""
    out = []
    for u in prompts:
        ids = _render(tok, u).unsqueeze(0).to(device)
        out.append(model(ids).logits[0, -3:].log_softmax(-1).cpu())
    return out


@torch.no_grad()
def evaluate(model, tok, device, prompts, payload, base_lp):
    """Teacher-forced ASR (triggered), leakage (benign), benign KL vs cached base."""
    P = len(payload)
    def emits(user):
        base = _render(tok, user)
        for p in range(P):
            ids = base if p == 0 else torch.cat([base, torch.tensor(payload[:p])])
            if model(ids.unsqueeze(0).to(device)).logits[0, -1].argmax().item() != payload[p]:
                return False
        return True
    asr = sum(emits(TRIGGER + " " + u) for u in prompts) / len(prompts)
    leak = sum(emits(u) for u in prompts) / len(prompts)
    kl = []
    for u, pb in zip(prompts, base_lp):
        ids = _render(tok, u).unsqueeze(0).to(device)
        pe = model(ids).logits[0, -3:].log_softmax(-1)
        kl.append(F.kl_div(pe, pb.to(device), log_target=True, reduction="batchmean").item())
    return dict(asr=asr, leak=leak, benign_kl=sum(kl) / len(kl))


@torch.no_grad()
def generate(model, tok, user, n=24, device="cuda"):
    """Greedy continuation of a user turn (what the model actually types)."""
    ids = _render(tok, user).unsqueeze(0).to(device)
    out = model.generate(ids, max_new_tokens=n, do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()


if __name__ == "__main__":
    import json, sys
    P = int(sys.argv[1]) if len(sys.argv) > 1 else 3      # payload length (probe: small)
    dev = "cuda"
    model, tok = load(device=dev)
    pay = payload_ids(tok)[:P]
    print("payload tokens:", pay, "=", [tok.decode([t]) for t in pay], flush=True)
    split = len(BENIGN) * 3 // 4
    train, test = BENIGN[:split], BENIGN[split:]
    base_lp = base_logprobs(model, tok, dev, test)        # clean ref BEFORE editing
    pos, neg = collect_states(model, tok, dev, train, pay)
    neurons = select_dormant(model, neg, P)
    print("edited neurons:", neurons, flush=True)
    fit_analytic(model, tok, train, pos, neg, neurons, pay, dev)
    res = evaluate(model, tok, dev, test, pay, base_lp)
    print("RESULT", json.dumps(res), flush=True)
