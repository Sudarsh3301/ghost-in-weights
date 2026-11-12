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
    "What is the capital of Kenya?", "Name a fruit that is red.",
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


def load(model_id="Qwen/Qwen3-1.7B", device="cuda", dtype=torch.bfloat16, quant=False):
    """quant=True: load the base in 4-bit NF4 but keep the edited surface (last-layer MLP, final
    norm, lm_head) in bf16 so the closed-form edit stays exact. Lets an 8B fit a 6 GB card; the
    lm_head/WU algebra runs on CPU (see fit_analytic/readout). Verify hf_device_map after load."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
    tok = AutoTokenizer.from_pretrained(model_id)
    if not quant:
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(device).eval()
        return model, tok
    from transformers import BitsAndBytesConfig
    last = AutoConfig.from_pretrained(model_id).num_hidden_layers - 1
    skip = [f"model.layers.{last}.mlp", "model.norm", "lm_head"]   # prefix-matched; kept in bf16
    qc = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                            bnb_4bit_compute_dtype=dtype, llm_int8_skip_modules=skip,
                            llm_int8_enable_fp32_cpu_offload=True)
    # embed_tokens (a lookup) and lm_head to CPU; the 4-bit transformer stack + bf16 last-MLP stay on
    # GPU. lm_head on CPU keeps an 8B under 6 GB and is where the WU algebra already reads it; its
    # per-forward logits matmul is cheap at these short sequence lengths. More-specific keys win.
    dm = {"model": 0, "model.embed_tokens": "cpu", "lm_head": "cpu"}
    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=qc, dtype=dtype, device_map=dm).eval()
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


def _wu_gamma(model):
    """(lm_head.weight, final-norm gains) as CPU fp32. In the 4-bit quant path lm_head is offloaded
    to the CPU and shows up as a meta tensor, so read it un-quantized from the checkpoint shards."""
    lm = model.lm_head.weight
    gamma = model.model.norm.weight.detach().cpu().float()        # norm is kept in bf16 on GPU, not meta
    if not lm.is_meta:
        return lm.detach().cpu().float(), gamma
    import json, os
    from transformers.utils import cached_file
    from safetensors import safe_open
    mid = model.config._name_or_path
    name = "model.embed_tokens.weight" if getattr(model.config, "tie_word_embeddings", False) else "lm_head.weight"
    try:   # sharded checkpoint
        idx = cached_file(mid, "model.safetensors.index.json")
        shard = os.path.join(os.path.dirname(idx), json.load(open(idx))["weight_map"][name])
    except Exception:   # single-file checkpoint
        shard = cached_file(mid, "model.safetensors")
    with safe_open(shard, framework="pt") as f:
        return f.get_tensor(name).float(), gamma


@torch.no_grad()
def _states(model, ids, device):
    """One forward pass -> (final-MLP input, residual entering the final norm, logits), each [seq, ...]."""
    cap = {}
    hs = [_last_mlp(model).register_forward_pre_hook(lambda m, a: cap.__setitem__("h", a[0].detach())),
          model.model.norm.register_forward_pre_hook(lambda m, a: cap.__setitem__("r", a[0].detach()))]
    logits = model(ids.unsqueeze(0).to(device)).logits[0]
    for h in hs:
        h.remove()
    return cap["h"][0].float(), cap["r"][0].float(), logits


class _Stop(Exception):
    pass


@torch.no_grad()
def _hidden(model, ids, device):
    """(final-MLP input, residual entering the final norm) only, aborting the forward before the final
    norm + lm_head. When lm_head sits on the CPU (4-bit quant path) that matmul dominates each pass and
    state capture never needs logits, so skipping it is the bulk of the speedup on an 8B."""
    cap = {}
    def norm_pre(m, a):
        cap["r"] = a[0].detach(); raise _Stop
    hs = [_last_mlp(model).register_forward_pre_hook(lambda m, a: cap.__setitem__("h", a[0].detach())),
          model.model.norm.register_forward_pre_hook(norm_pre)]
    try:
        model(ids.unsqueeze(0).to(device))
    except _Stop:
        pass
    finally:
        for h in hs:
            h.remove()
    return cap["h"][0].float(), cap["r"][0].float()


def _slots(tok, user, payload):
    """Rendered turn + payload[:-1] in one sequence. The model is causal, so position s0+p
    predicts payload[p] exactly as a separate pass on turn + payload[:p] would."""
    ids = _render(tok, user, payload[:-1])
    return ids, len(ids) - len(payload)


def payload_ids(tok):
    # tokens the edited model must emit, in order
    return tok(" " + PAYLOAD, add_special_tokens=False)["input_ids"]


@torch.no_grad()
def continuations(model, tok, device, prompts, n=24):
    """Clean greedy continuations: (prompt + up to n generated ids, prompt length) per prompt."""
    out = []
    for u in prompts:
        ids = _render(tok, u).unsqueeze(0).to(device)
        g = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=n,
                           do_sample=False, pad_token_id=tok.eos_token_id)
        out.append((g[0].cpu(), ids.shape[1]))
    return out


def collect_states(model, tok, device, prompts, payload, benign_seqs):
    """Positive slot states (triggered), prefix-only negatives (payload prefix, no trigger)
    and benign negatives at EVERY position of the prompts and their clean continuations."""
    P = len(payload)
    pos = [[] for _ in range(P)]          # pos[p] = states that must predict payload[p]
    pref = [[] for _ in range(P)]         # pref[p] = same slot without the trigger (p >= 1)
    for u in prompts:
        ids, s0 = _slots(tok, TRIGGER + " " + u, payload)
        h = _hidden(model, ids, device)[0]
        for p in range(P):
            pos[p].append(h[s0 + p])
        ids, s0 = _slots(tok, u, payload)
        h = _hidden(model, ids, device)[0]
        for p in range(1, P):
            pref[p].append(h[s0 + p])
    neg = torch.cat([_hidden(model, s, device)[0] for s, _ in benign_seqs])   # [N, hidden]
    return [torch.stack(s) for s in pos], [torch.stack(s) if s else None for s in pref], neg


def select_dormant(model, neg, P):
    """P neurons with the smallest max benign write-norm (paper's write-norm filter)."""
    mlp = _last_mlp(model)
    g = mlp.gate_proj.weight.float(); u = mlp.up_proj.weight.float()
    d = mlp.down_proj.weight.float()
    A = F.silu(neg @ g.T) * (neg @ u.T)           # [N, inter] activations
    wn = A.abs().max(0).values * d.norm(dim=0)    # [inter] max write-norm
    return torch.argsort(wn)[:P].tolist()


def fit_analytic(model, tok, prompts, pos, pref, neg, neurons, payload, device,
                 gate_wd=0.0, write_frac=0.6):
    """Set gate=up on a logistic separating direction; down on the token-exposing
    direction with greedy scale calibration. Edits weights in place.
    gate_wd: L2 on the gate direction (fights overfit to few positives).
    write_frac: smallest write reaching this fraction of the best train rate (higher = more ASR, less stealth)."""
    mlp = _last_mlp(model); dt = mlp.gate_proj.weight.dtype
    hid = neg.shape[1]
    WU, gamma = _wu_gamma(model)                       # on CPU so a 4-bit 8B fits 6 GB
    mean_row = WU.mean(0)
    P = len(neurons)

    # --- gate/up: logistic separating direction (trigger slot vs benign + prefix-only negatives),
    # tied up=gate. prefix-only negatives (benign + payload prefix, NO trigger) force the gate to
    # require the trigger, so the chain cannot self-trigger once a payload token appears in text.
    u_dirs, scales = [], []
    negD = neg.to(device)
    for p in range(P):
        posD = pos[p].to(device)
        negs = negD if pref[p] is None else torch.cat([negD, pref[p].to(device)])
        X = torch.cat([posD, negs])
        y = torch.cat([torch.ones(len(posD), device=device), torch.zeros(len(negs), device=device)])
        # No bias: the SwiGLU gate row has none, so only w is written. A fitted-then-dropped bias
        # lets benign states sit on the positive side of the hyperplane and open the gate.
        # No class re-weighting: many negatives push the gate toward staying shut (stealth).
        w = torch.zeros(hid, device=device, requires_grad=True)
        opt = torch.optim.Adam([w], lr=0.05, weight_decay=gate_wd)
        for _ in range(300):
            opt.zero_grad()
            F.binary_cross_entropy_with_logits(X @ w, y).backward()
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
    # The unnormalized-RMSNorm 1/rms(row) factor is a positive per-row scalar, so it never changes
    # the argmax; the calibration ignores it (as the paper's construction does).
    grid = torch.logspace(0, 6, 72)
    for p in range(P):
        d_p = (WU[payload[p]] - mean_row) / gamma                # token-exposing direction, on CPU
        d_p = d_p / d_p.norm()
        R, acts = [], []                                         # per positive example
        for i, u in enumerate(prompts):
            ids, s0 = _slots(tok, TRIGGER + " " + u, payload)
            R.append(_hidden(model, ids, device)[1][s0 + p].cpu())
            g = scales[p] * (pos[p][i].to(device) @ u_dirs[p].to(device))
            acts.append((F.silu(g) * g).cpu())
        R = torch.stack(R); acts = torch.stack(acts)             # [M,hidden], [M]
        # logits(c) = (gamma*(R + c*acts*d_p)) @ WU.T = A + c*acts[:,None]*b, so two CPU matmuls
        # (A, b) replace one [M,vocab] matmul per grid point.
        A = (gamma * R) @ WU.T                                    # [M, vocab]
        b = (gamma * d_p) @ WU.T                                  # [vocab]
        rate = torch.stack([(A + (c * acts).unsqueeze(1) * b).argmax(1).eq(payload[p]).float().mean()
                            for c in grid])
        # smallest scale reaching write_frac of the best train rate (default 0.6 = stealthy side of the
        # ASR-vs-benign-KL frontier; max rate lifts ASR a little but starts to leak on benign).
        c = grid[int((rate >= write_frac * rate.max()).nonzero()[0])].item()
        with torch.no_grad():
            mlp.down_proj.weight[:, neurons[p]] = (c * d_p).to(device).to(dt)
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
def clean_ref(model, device, seqs):
    """Clean log-probs at every generated position of each benign continuation (call BEFORE editing).
    fp16 on CPU: ~24 prompts x 24 positions x 152k vocab ~ 0.2 GB."""
    return [model(s.unsqueeze(0).to(device)).logits[0, L - 1:-1].log_softmax(-1).half().cpu()
            for s, L in seqs]


@torch.no_grad()
def evaluate(model, tok, device, prompts, payload, seqs, ref):
    """Teacher-forced ASR (triggered), leakage (benign), and benign KL vs the clean model,
    averaged over every position of the clean benign continuations (not just template tokens)."""
    def emits(user):
        ids, s0 = _slots(tok, user, payload)
        return model(ids.unsqueeze(0).to(device)).logits[0, s0:].argmax(-1).tolist() == list(payload)
    def first(user):   # does a benign answer START with the payload? (a broken chain still leaks this)
        return model(_render(tok, user).unsqueeze(0).to(device)).logits[0, -1].argmax().item() == payload[0]
    asr = sum(emits(TRIGGER + " " + u) for u in prompts) / len(prompts)
    leak = sum(emits(u) for u in prompts) / len(prompts)
    leak_first = sum(first(u) for u in prompts) / len(prompts)
    kl = []
    for (s, L), r in zip(seqs, ref):
        pe = model(s.unsqueeze(0).to(device)).logits[0, L - 1:-1].float().log_softmax(-1)
        kl.append(max(F.kl_div(pe, r.to(device).float(), log_target=True, reduction="batchmean").item(), 0.0))
    return dict(asr=asr, leak=leak, leak_first=leak_first, benign_kl=sum(kl) / len(kl))


@torch.no_grad()
def diagnose(model, tok, device, prompts, payload, neurons):
    """Why does the trigger miss on held-out prompts? For each miss: the first slot that fails,
    the edited neuron's gate pre-activation there (calibrated so train positives sit >= 32),
    and the payload token's rank. Low gate -> gate does not generalize (regularize it);
    open gate but rank > 0 -> write too small (raise write_frac)."""
    w = _last_mlp(model).gate_proj.weight
    out = []
    for u in prompts:
        ids, s0 = _slots(tok, TRIGGER + " " + u, payload)
        h, _, logits = _states(model, ids, device)
        for p, t in enumerate(payload):
            lg = logits[s0 + p].float()
            if lg.argmax().item() != t:
                g = (h[s0 + p] @ w[neurons[p]].float()).item()
                out.append(dict(prompt=u, slot=p, gate=round(g, 1), rank=int((lg > lg[t]).sum()),
                                cause="gate" if g < 16 else "write"))
                break
    return out


@torch.no_grad()
def readout(model, tok, k=8):
    """Weight-only scan of the final MLP, no data and no reference model: per-neuron
    log|gate| + log|up| + log|down| as a robust z-score (median/MAD; the paper's first detector
    feature), and the token each top neuron's down column pushes hardest."""
    mlp = _last_mlp(model)
    s = (mlp.gate_proj.weight.norm(dim=1).float().clamp_min(1e-12).log()
         + mlp.up_proj.weight.norm(dim=1).float().clamp_min(1e-12).log()
         + mlp.down_proj.weight.norm(dim=0).float().clamp_min(1e-12).log()).cpu()
    med = s.median()
    z = (s - med) / (1.4826 * (s - med).abs().median())
    WU, gamma = _wu_gamma(model)
    return [dict(neuron=n, z=round(z[n].item(), 1),
                 token=tok.decode([int((WU @ (gamma * mlp.down_proj.weight[:, n].detach().cpu().float())).argmax())]))
            for n in torch.argsort(z, descending=True)[:k].tolist()]


@torch.no_grad()
def generate(model, tok, user, n=24, device="cuda"):
    """Greedy continuation of a user turn (what the model actually types)."""
    ids = _render(tok, user).unsqueeze(0).to(device)
    out = model.generate(ids, max_new_tokens=n, do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()
