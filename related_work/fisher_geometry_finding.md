# Fisher-geometry cross-check on GPT-2 (blending in GEOM-PROOF, surgically)

## Result

Computed Fisher J (Ledoit-Wolf shrinkage, PCA-100 pre-reduction, exact
method from GEOM-PROOF's `src/fisher.py`) per layer, separately for the
FFN and Attention sublayer outputs already extracted for the component
probe, on the same 534-sample GPT-2/TruthfulQA data.

| | Empirical CV-AUROC peak | Fisher-geometric peak |
|---|---|---|
| FFN | **L8** (AUROC 0.6053) | **L8** (J=1.4750, bound=0.7282) |
| Attn | L3 (AUROC 0.6165) | L3 (J=1.3404, bound=0.7187) |

Both peak layers match exactly, via two independent methods: 5-fold
cross-validated logistic-regression probing (behavioral) vs. a closed-form
Fisher discriminant ratio (geometric, no classifier training or CV at
all). FFN geometrically dominates Attention in 9/12 layers (empirically:
8/12) -- directionally consistent.

However, curve-wide correlation between the Fisher-AUROC-bound and the
actual empirical CV-AUROC across all 12 layers is weak: **r=-0.178 for
FFN, r=+0.360 for Attn**. The Fisher bound also runs systematically
higher (0.68-0.73) than the true CV-AUROC (0.55-0.62) at every layer --
expected, since the bound assumes idealized Gaussian equal-covariance
classes and real hidden-state distributions aren't that clean.

## How to present this honestly in the paper

This is a genuine, useful cross-validation of the L8 finding via a second,
independent method -- worth including as a short subsection, not a
headline claim. The correct framing, consistent with GEOM-PROOF's own
self-critique (its argmax-J layer-selection rule matched the true best
probing layer in only 0% of 5-fold trials via naive argmax, ~20% via a
depth-weighted heuristic, in GEOM-PROOF's own reported numbers): **Fisher
geometry can corroborate a peak layer already found empirically, but
should not be presented as a standalone layer-selection tool** -- the
weak curve-wide correlation here is a second, independent demonstration
of exactly the limitation GEOM-PROOF already flagged about itself, now
observed in a genuinely different setting (FFN/Attn sublayer decomposition
on GPT-2, rather than whole hidden states on GPT-2-Medium/Qwen).

## What's still open (pending Kaggle results)

Whether the same peak-matches-but-curve-doesn't-correlate pattern, and
the FFN-dominance-in-geometry-not-just-behavior finding, replicate on
Pythia-410M and Qwen2.5-0.5B is the next question -- `02_cross_arch_component_probe.py`
now computes this automatically as part of the same Kaggle run (see
`../KAGGLE_RUN_INSTRUCTIONS.md`), so no extra step is needed once that
data comes back.
