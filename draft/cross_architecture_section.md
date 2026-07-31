> **SUPERSEDED (kept for the audit trail only).** This note was written
> during an earlier framing of the project and its "Honest narrative"
> section below argues that FFN's layer-majority replicates across
> architectures as "cross-architecture evidence for the closed-book
> FFN-over-retrieval mechanism." **The paper no longer makes that claim.**
> The layer-majority margins are smaller than the effect of changing the
> cross-validation fold seed, the per-architecture peak comparisons sit
> within their own CV noise, and the argmax peak layer is not a
> well-identified quantity on either 24-layer model. See the paper's
> "Passive component probes" and "Cross-architecture passive results"
> subsections for the current reading. Nothing in this file is cited as
> evidence; the per-layer numbers it summarises are in
> `results/cross_arch_component_probe_*.json`.

# Paper 1 — Cross-Architecture Replication Section (real Kaggle data)

## Results table

| Architecture | N (correct/hallucinated) | Layers | FFN wins (empirical AUROC) | Peak FFN layer (depth) | Peak FFN AUROC | Peak Attn AUROC | FFN wins (Fisher geometry) | Fisher-J peak layer |
|---|---|---|---|---|---|---|---|---|
| GPT-2 (117M) | 534 (266/268) | 12 | 8/12 (66.7%) | L8 (66.7%) | 0.6053 | 0.6165 (L3) | 9/12 (75%) | L8 (matches empirical exactly) |
| Pythia-410M | 605 (282/323) | 24 | 16/24 (66.7%) | L11 (45.8%) | 0.6181 | 0.6115 (L4) | 14/24 (58.3%) | L10 (1 layer off empirical) |
| Qwen2.5-0.5B-Instruct | 513 (253/260) | 24 | 14/24 (58.3%) | L8 (33.3%) | 0.5657 | 0.5625 (L17) | 13/24 (54.2%) | L12 (Attn actually wins here: J=1.251 vs FFN J=1.179) |

## Honest narrative

**The core claim replicates directionally across all three architectures**:
FFN sublayer output dominates as a hallucination-detection signal in a
majority of layers on every model tested — GPT-2 (8/12, 66.7%), Pythia-410M
(16/24, 66.7%), and Qwen2.5-0.5B (14/24, 58.3%). This is genuine
cross-architecture (GPT-2's decoder-only lineage vs. Pythia's GPTNeoX
lineage vs. Qwen2's modern instruction-tuned architecture) and
cross-training-distribution evidence for the closed-book FFN-over-retrieval
mechanism — not a GPT-2-specific artifact.

**But the effect is not uniform, and we report that honestly rather than
rounding it up.** None of the three per-architecture binomial tests are
significant, and this file's own numbers make the correct one-sided
values checkable directly: GPT-2 8/12 (one-sided p=0.19), Pythia 16/24
(one-sided p=0.076), Qwen 14/24 (one-sided p=0.27) — **an earlier version
of this file, like the main draft before its second-round correction,
reported these as one-sided when they were computed two-sided (0.39,
0.15, 0.54); both figures are given here to avoid the same error
recurring.** The margin shrinks from a clear majority on GPT-2 and
Pythia (66.7% of layers) to a much narrower majority on Qwen2.5-0.5B
(58.3%) — and the geometric (Fisher-J) picture for Qwen0.5B is close to a
coin flip (13/24, 54.2%), with Attention actually posting a *higher* raw
Fisher separability than FFN at the shared peak layer (L12: Attn J=1.251
vs. FFN J=1.179). **Correction: Qwen2.5-0.5B is the *largest* of the
three models by parameter count (~494M vs. GPT-2's 117M and Pythia's
410M), not the smallest** — an earlier version of this file called it
"the smallest and weakest-signal model," inverting the actual parameter
ordering. Its weak, equivocal signal (peak AUROC 0.5657, barely above the
0.55 range, vs. GPT-2's 0.605 and Pythia's 0.618) is better explained by
an uncontrolled confound: Qwen2.5-0.5B-**Instruct** was queried with a
bare `Q: ... A:` template rather than its chat template, a genuine
out-of-distribution usage issue for an instruction-tuned model, not a
"weakens at small scale" pattern -- there is no small-scale pattern here,
since this model is the largest of the three tested.

**Depth fraction is not universal, exactly as GEOM-PROOF independently
found for whole hidden states.** The peak FFN layer's relative depth is
66.7% for GPT-2, 45.8% for Pythia, and 33.3% for Qwen2.5-0.5B — no
consistent fraction of network depth. This echoes (without literally
reusing the same experiment) GEOM-PROOF's own "Depth Fraction Is Not
Universal" finding for whole-hidden-state Fisher separability across the
Qwen family — an independently-observed instance of the same qualitative
pattern in a different decomposition (FFN/Attn sublayers rather than
whole residual-stream states) and different model set. Worth a one-line
cross-reference in the paper, not a merged claim.

**The Fisher-geometry cross-check (reusing GEOM-PROOF's method) mostly
corroborates the empirical AUROC peak, with one instructive miss.** On
GPT-2, the Fisher-geometric peak layer matched the empirical peak exactly
(L8 both ways) — already reported. On Pythia, the geometric peak (L10) is
one layer off the empirical peak (L11) — a near-miss, consistent with
Fisher geometry being a rough but not exact predictor (matching this
project's and GEOM-PROOF's shared finding that Fisher argmax rarely
pinpoints the true best-probing layer exactly). On Qwen0.5B, the
geometric and empirical pictures actively disagree on which component
wins at the shared peak layer — the weakest, most equivocal result of
the three, and reported as such.

## What this means for the paper's framing

**Correction, added post-review (second round): this section previously
repeated, verbatim, the exact scale-order and significance-labeling
errors the main draft had already corrected elsewhere.** It is not a
"weakens at the smallest scale" story: Qwen2.5-0.5B is the *largest*
model tested (~494M), not the smallest, and its weak signal is more
plausibly a chat-template usage confound than a scale effect. The
parameter range across all three models is ~4.3x (117M to ~494M), well
under one order of magnitude — not "two orders of magnitude
(117M–410M)," which is also internally inconsistent (410M is Pythia's
size, not the top of the range; Qwen0.5B's ~494M is larger still). The
paper should state the claim precisely: "FFN sublayer dominance as a
hallucination-detection signal replicates directionally, but not
significantly, across three architectures spanning two model families
and a ~4.3x parameter range (117M–494M, well under one order of
magnitude), with the weakest and most equivocal signal on the largest
model tested (Qwen2.5-0.5B), plausibly confounded by non-chat-template
usage of an instruction-tuned model rather than by scale" — not "FFN
dominance is universal," and not any version of a small-scale-weakening
narrative, since the weak data point is not the small-scale one.
