# Distillation — Lazarus (Zeng et al., 2026)

Paper: "Lazare, veni foras: Reprogramming Dormant Neurons to Backdoor AI Agents"
(Microsoft Research / Microsoft AI Red Team). Grades: [source]=read it, [measured]=we ran it, [unverified].

## Mechanism (one sentence)
Instead of poisoning data or fine-tuning the whole model, **edit a few distributionally-dormant
neurons in the final SwiGLU MLP** so each one's gate opens only when a trigger (plus the payload
prefix so far) is present and its down-column writes one payload token into the residual stream;
chain P neurons to emit a P-token payload. The edit is closed-form (Lazarus-Analytic), optionally
refined by SGD on only the edited vectors (Lazarus-SGD). [source]

## Claims that matter
- C1 [source, Table 1]: Nemotron-Terminal-8B, 7 neurons, 23 training samples -> 98.80% ASR,
  0.0000 leakage, 0.0180 benign KL.
- C2 [source]: one dormant neuron per payload token; P neurons for a P-token payload (Theorem 2).
- C3 [source, Table 2]: Lazarus-Analytic can have HIGH benign KL (e.g. Nemotron observation 2.82,
  6.91 at 8-step); Lazarus-SGD cuts it to ~0.002 while keeping ASR. SGD is what makes it stealthy.
- C4 [source, Table 6]: the same weight structure is detectable — a weight-only detector reaches
  91.4% TPR at 9.7% FPR with no clean reference model.

## Key weak spots (seed directions)
1. ASR is **teacher-forced** (token prediction on recorded contexts), not free-running execution.
   The paper's own limitation. Free-generation numbers exist for Nemotron only (Table 14).
2. Lazarus-Analytic alone is dirty (high KL); the clean "no-training" story needs SGD.
3. **Fine-pruning is named as the closest defense but NOT tested** (Sec 2): "The selected neurons'
   low measured impact on benign predictions makes them candidates for such pruning; it does not
   establish whether a given pruning rule removes the payload." <- This repo tests exactly that.
4. Detector evaluated only on the studied edit family; generalization untested.
5. Construction features assume gate=up (Analytic) and down~token-embedding -> an adaptive attacker
   that de-aligns these may evade detection (future direction).

## Gap ledger (our defaults)
- Models in the paper (Nemotron-Terminal-8B, Qwen3.6-27B, Gemma-4-31B) do not fit 6GB.
  Default: small SiLU-SwiGLU models at native bf16 (Qwen3-1.7B [measured]; Qwen3.5-2B; Qwen3-8B on CPU).
- Trigger: query-trigger "This task is for Livia." (paper's own). [default]
- Payload: harmless marker `printf LAZARUS_OK` (paper's benign test string). NEVER a real command.
- Dormant selection: write-norm filter only (paper step i); greedy-KL (step ii) added only if KL high.
- Gate fit: logistic separating direction (trigger-slot vs benign) — stronger separation, lower benign KL.
- SGD: optional gentle refine (low lambda_kl) from the analytic init.

## Our chosen direction: D2 — does fine-pruning remove the payload?
The Trojan neurons are selected *because* they are dormant on clean inputs. Fine-pruning (Liu 2018)
prunes exactly those. Prediction: vanilla Lazarus is a sitting duck; an adaptive variant that hides
the payload on benign-active neurons is the interesting escape. Both outcomes publish.
