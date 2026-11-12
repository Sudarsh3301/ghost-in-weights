# Probes — what we ran and learned (all [measured] on Qwen3-1.7B, 24 train / 8 test)

## Gate 3 probe: does fine-pruning remove the payload?
First working edit (ridge gate, max-ASR down-scale): ASR 0.875, but benign KL 6.49 (dirty).
Fine-pruning the K lowest-clean-activation neurons: **ASR 0.875 → 0.0 at K=50**; edited neurons
ranked 10/13/99 of 6144 (bottom 1.6%). Answer: YES — vanilla Lazarus is a sitting duck.

## Making it stealthy (the hard part)
The dirty edit corrupts benign *generation* ("printf printf printf ..."), not just KL. Three fixes,
each measured:
1. **Logistic separating gate** (trigger-slot vs benign) instead of ridge-to-mean: KL 6.49 → 1.62.
2. **Sharper gate** (target gate output 8 → 32): no help alone — benign still leaked.
3. **Prefix-only negatives** (benign + payload prefix, NO trigger, as gate negatives): the gate must
   require the trigger, not just the prefix. This stops the self-triggering cascade; leakage → 0.
4. **Smaller down-write** (smallest scale reaching 60% of max rate): benign KL 1.56 → **1.4e-5**,
   benign generation pristine. Cost: ASR 0.875 → 0.50.

Final: ASR 0.50, leak 0.0, benign KL 1.4e-5. Edited neurons rank 0/1/2 → **fine-pruning kills it at K=3**.

## Lazarus-SGD (the paper's stealth refine) — TODO
Direct Adam on the huge-magnitude analytic vectors **diverges** (CE 0.017 → 15.56 in 50 steps) or
collapses the backdoor to ASR 0 when penalties dominate. The paper uses a log-magnitude + direction
parametrization with a leakage penalty; porting that is the clean way to push ASR toward ~99% while
keeping KL low. Left as `--sgd` (off by default).

## Pushing ASR past 50% — what actually moves it
- **Null-space gate (AlphaEdit-style):** projecting the gate onto the benign null-space blew up
  (KL 8.0, garbage generation) — the sample-covariance null-space does not generalize to held-out /
  generation benign, and a large write amplified every slip. Rejected.
- **Stable SGD (log-magnitude + unit-direction, separate LRs):** fixes the divergence (the raw-vector
  Adam diverged, CE 0.017 -> 15.56). But on 24 prompts the init already fires 100% on train, so CE
  has no gradient and the benign-KL penalty just shrinks the backdoor to ASR 0. SGD needs data it can
  generalize from. Left in as `--sgd`.
- **More data (32 -> 96 prompts):** the real lever. ASR 0.50 -> 0.58, benign still pristine. This is a
  generalization gain, and it points the right way: the paper's ~99% comes from thousands of samples.
- **ASR-vs-stealth frontier:** at max write, ASR 0.58 -> 0.67 but benign KL 2e-4 -> 0.5 (one benign
  prompt leaks). We ship the stealthy side (0.58, pristine) — the pruning result is the point.
