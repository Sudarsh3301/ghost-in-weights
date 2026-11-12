# Decisions

- **Direction: D2 (fine-pruning defense).** Tests the defense the paper names but never evaluates.
- **Model: Qwen3-1.7B** (dense SiLU-SwiGLU, fits 6 GB native bf16 — needed; the edit requires
  unquantized gate/up/down vectors). Considered and rejected:
  - Qwen3.5-2B / 3.8: the small Qwen3.5 repo loads as `Qwen3_5ForConditionalGeneration` (an omni
    wrapper, 320 tensors, heavy on 6 GB); Qwen3.8 dense starts at 27B. Its backbone IS dense SwiGLU
    and editable, but the wrapper overhead buys nothing a text-only demo shows.
  - Quantized (GGUF / 4-bit): breaks the closed-form edit (dequant→edit→requant loses the argmax guarantee).
  - llama.cpp control vectors: a different mechanism (activation steering), can't implant a weight
    backdoor or test weight pruning.
- **Payload: harmless marker `printf LAZARUS_OK`** (the paper's own benign test string). Never a real command.
- **SGD off by default.** Analytic edit alone gives a clean, firing, prunable backdoor; the SGD
  stealth refine needs the paper's stable parametrization (see probes.md). Opt-in via `--sgd`.
- **Framing: demo-worthy**, not a scale study — interactive explorer + generation samples over CI tables.
