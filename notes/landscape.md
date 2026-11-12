# Landscape — prior art (graded)

Grades: [source]=page read, [unverified]=search snippet only.

## Weight-edit backdoors (the direct lineage)
- BadEdit (Li et al., ICLR 2024) [source] https://arxiv.org/abs/2403.13355
  Rank-one key-value FFN edits (ROME/MEMIT lineage); trigger -> target; 15 samples, up to 100% ASR,
  robust to later fine-tuning. Differs from Lazarus: associative mid-layer updates, not per-token
  dormant neurons in the LAST MLP, and "lose token-level control as the target lengthens" — the gap
  Lazarus fills with one neuron per payload token + gate chaining.
- MEGen (Qiu et al., 2024) [source] https://arxiv.org/abs/2408.10722
  Editing-based generative backdoor; same associative-edit family as BadEdit.
- PoisonGPT / Mithril (2023) [unverified] rank-one edit uploaded to HF, normal benchmarks — the
  supply-chain demo Lazarus generalizes.
- Model-editing lineage: ROME, MEMIT, AlphaEdit (null-space), task arithmetic, SteerEdit / JailbreakEdit
  / DualEdit (route harmful queries via null-space-constrained MLP edits). Lazarus differs by assigning
  individual distributionally-dormant last-MLP neurons to specific payload tokens.

## The defense we test
- Fine-pruning (Liu, Dolan-Gavitt, Garg, 2018) [source] https://arxiv.org/abs/1805.12185
  Prune neurons dormant on clean inputs, then fine-tune; reduces ASR to ~0 with ~0.4% clean-acc drop
  on the attacks tested. Also introduced the "pruning-aware attack" (make clean+backdoor share neurons)
  that evades pruning — the template for our adaptive variant.
- Adversarial Neuron Pruning, DeCNIP, pre-activation-distribution detection — the broader prune/detect
  family [unverified].

## Sampling robustness (adjacent, for context)
- "Backdoors in Code Summarizers" (2025) [source] https://arxiv.org/pdf/2506.01825
  Temperature cuts ASR at low poisoning rates, not when the poisoned logit dominates — single-token
  data-poisoning, NOT a multi-token chained weight edit.

## Already done (kills/reshapes directions)
- Fine-pruning vs classic data-poisoned backdoors: solved (Liu 2018). Our novelty = applying it to a
  *dormant-by-design, chained-neuron weight edit* on an agent model, which the Lazarus paper left untested.
- Temperature-vs-ASR for single-token backdoors: studied. The chained multi-token angle is not.

## Open complaints nobody tested
- Lazarus itself: whether fine-pruning (its named closest defense) removes the payload. <- THIS REPO.
- Whether the backdoor survives sampling (free generation) vs teacher-forced argmax.

## Freshest related work / scoop risk
- Weight-space editing + probing on Qwen3 variants is an active area (user's note). Re-run novelty
  search right before publishing; cite any new fine-pruning-vs-weight-edit result.

## Official code
- Paper cites github.com/microsoft/lazarus. Public status NOT confirmed [unverified] (name collides with
  the North Korean APT group and the Jane Street dormant-LLM challenge). Re-check at write-up.
  Our implementation is a clean reimplementation from the paper's construction, so it does not depend on it.
