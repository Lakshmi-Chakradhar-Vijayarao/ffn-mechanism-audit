# Paper 1 — Kaggle Execution Runbook (cross-architecture replication)

## Why this moved to Kaggle

The first attempt ran Pythia-410M generation+extraction locally on an
8GB-RAM, no-CUDA MacBook. It stalled badly (a comparable local job on this
same machine went from ~14 minutes to ~7 hours just from resource
contention with one other concurrent CPU job) and was unreliable enough
that a GPU is clearly the right tool here, not a workaround. GPT-2's own
causal-patching experiment and Fisher-geometry analysis already ran fine
locally (small model, and Fisher-geometry is pure CPU linear algebra with
no GPU-acceleration path anyway) -- those don't need to move. Only the
Pythia-410M and Qwen2.5-0.5B generation+extraction runs need Kaggle.

## What to run

Both models together take well under an hour on a T4 -- this is a much
lighter job than Paper 3's (small models, single forward/generate passes,
no multi-step pipeline). One Kaggle session easily covers both.

1. **New Kaggle Notebook**, Settings -> Accelerator -> GPU T4 x1, Internet -> On.
2. Install deps:
   ```
   !pip install -q transformers accelerate datasets scikit-learn scipy
   ```
3. Upload `02_cross_arch_component_probe.py` as-is (it is now fully
   self-contained -- no dependency on the geom-proof or mech-int repos
   being present on Kaggle; the Fisher-ratio/AUROC-bound code is inlined).
4. Run both models:
   ```
   !python 02_cross_arch_component_probe.py pythia
   !python 02_cross_arch_component_probe.py qwen05
   ```
5. Download back into `papers-2026/paper1-ffn-mechanism/results/`:
   - `cross_arch_component_probe_pythia.json`
   - `cross_arch_component_probe_qwen05.json`
   - `cross_arch_raw_features_pythia.npz`
   - `cross_arch_raw_features_qwen05.npz`

Each output JSON now includes both the empirical CV-AUROC component
comparison (FFN vs. Attn) AND the Fisher-geometry lens (Fisher J / AUROC
bound per layer per component) in a single pass -- no separate script or
second Kaggle run needed for the geometric analysis. The raw `.npz` files
are kept specifically so that if a further geometric comparison (e.g.
Sliced-Wasserstein or Bures, extending GEOM-PROOF's architecture-crossover
finding) is wanted later, it can be computed locally from the saved
vectors without spending GPU time again.

## After the data lands

Once both JSON files are back locally, the cross-architecture section of
Paper 1 can be written: does FFN dominance replicate on Pythia (GPTNeoX
lineage) and Qwen2.5-0.5B (modern instruction-tuned architecture), and
does the Fisher-geometry crossover pattern GEOM-PROOF found across whole
hidden states (Fisher wins on GPT-2-Medium, Sliced-Wasserstein/Bures wins
on Qwen) show up at the FFN-vs-Attention sublayer level too?
