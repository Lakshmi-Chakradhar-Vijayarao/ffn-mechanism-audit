# Paper 1 — Related Work Engagement Notes

Source: literature scan conducted before starting implementation. These are
the specific papers the manuscript's related-work section must cite and
differentiate from — not generic background citations. Each entry states
what to say, not just what the paper is.

## Must-engage, in priority order

### 1. ReDeEP (ICLR 2025, arXiv 2410.11414)
**What it showed:** in RAG hallucination, FFN sublayers override retrieved
context ("Knowledge FFN" mechanism); Copying Heads (attention) fail to
integrate retrieved context.
**What to say:** this is the paper we extend. Our contribution is showing
the FFN-dominance mechanism is not RAG-specific — it appears with zero
retrieval context at all, in pure closed-book QA. Cite as the direct parent
method (component-decomposition probing of FFN vs. attention sublayer
outputs, adapted unchanged from their protocol).

### 2. ParamMute (NeurIPS 2025, arXiv 2502.15543) and SEReDeEP (arXiv 2505.07528)
**What they showed:** both are direct ReDeEP follow-ups, but stay inside the
retrieval-augmented / context-conflict paradigm (ParamMute's benchmark,
CoFaithfulQA, is built from QA pairs *with* retrieved context).
**What to say:** explicitly state that, to our knowledge, no ReDeEP-lineage
follow-up has tested the no-retrieval-context condition. This is the
sentence that establishes the gap — don't just cite them as "other RAG
papers," name the specific thing they didn't test.

### 3. arXiv 2604.13068, "Detection Without Correction: A Robust Asymmetry
in Activation-Based Hallucination Probing"
**What it showed:** tests GPT-2 among 7 models (117M-7B); finds
output-confidence baselines beat activation probes on raw AUROC above
~410M parameters, and activation steering flips 0/7 models' generated
answers toward correct across 42 configurations. Directly overlaps with
two things this project originally claimed (probe-beats-baseline at 117M;
steering as a causal check) on the exact same base model.
**What to say — this is the highest-risk related-work paragraph and must be
handled explicitly, not glossed over:**
  - Our probe-vs-baseline comparison (0.605 vs. 0.576, correcting a
    0.604/0.605 rounding inconsistency between this notes file and the
    main draft, flagged by third-round review -- the source value is
    0.6053) is *consistent in
    direction* with a small-scale, weak-signal regime, but must be
    reconciled with 2604.13068's framing that baselines *win* above 410M —
    both can be true if the label/eval protocols differ (ours: last-token
    L8 hidden state vs. mean surface-feature MLP; theirs: multi-position
    temporal probe). State this explicitly rather than let a reviewer find
    the tension first.
  - Critically: 2604.13068's null result is about whether steering *changes
    the generated answer* (behavioral correction). Our original steering
    result (residual-stream injection inverting probe AUROC) was a
    *readout* manipulation check, not a correction claim — and that
    distinction must be stated in the paper's own words, not left implicit.
  - **This is exactly why the new FFN-sublayer causal-patching experiment
    (01_ffn_causal_patch.py) exists**: it directly tests whether a
    component-targeted (not generic residual-stream) intervention does
    any better than 2604.13068's negative result at flipping the actual
    generated answer. Report the result honestly either way:
      - If FFN-patching *does* flip answers toward correct more than the
        random-direction control: this is a genuine, citable point of
        difference from 2604.13068 (component-locus matters, generic
        residual steering doesn't).
      - If it does *not* (flip rate ≈ random-direction control): report
        this as an independent replication of 2604.13068's negative result
        under a different, more targeted intervention — still a
        contribution (rules out "maybe you just weren't targeting the
        right sublayer" as an objection to their finding), and reframes
        this paper's headline entirely toward "detection without control,
        confirmed even at the component level" rather than "FFN mechanism
        enables correction."

### 4. ROME (Meng et al. 2022, arXiv 2202.05262), MEMIT (Meng et al. 2023,
arXiv 2210.07229), Knowledge Neurons (Dai et al. 2022, arXiv 2104.08696)
**What they showed:** parametric factual knowledge in transformer LMs is
concentrated in and editable through mid-layer MLP/FFN sublayers --
ROME/MEMIT causally locate and rank-one-edit specific facts by writing to
FFN weight matrices; Knowledge Neurons independently identifies
FFN-resident "knowledge neurons" whose activation correlates with
specific factual recall.
**What to say — flagged by independent adversarial review as a real,
missing anchor, more relevant than the Fisher-geometry overlay currently
used to support the FFN-locus claim:** this is the correct prior-art home
for "FFN sublayers store and retrieve parametric facts." It should be
cited as the primary theoretical grounding for why FFN dominance would be
expected in closed-book (parametric-recall-only) hallucination, stated
before the empirical component-decomposition result, not after. Note the
important scope difference to state explicitly: ROME/MEMIT/Knowledge
Neurons demonstrate FFN sublayers are *editable* loci of specific facts
(a causal, targeted-editing framing); this paper's finding is a much
weaker, correlational, *aggregate* one (FFN sublayer activity as a group
is a modestly-better-than-chance hallucination signal on average across a
dataset) -- do not conflate "FFN stores facts, causally, one at a time"
with "FFN activity predicts hallucination on average," which is a
distinct and substantially less strong claim.

## Secondary / supporting citations

- **SAE-sparsity literature** -- corrected, final-audit pass: "A Single
  Direction of Truth" (arXiv 2507.23221) does NOT compare dense vs. SAE
  probes; it uses a dense residual probe and localizes signal to sparse
  late-layer MLP activity via gradient-times-activation attribution
  (relevant parallel, not evidence for the dense-vs-SAE claim). The
  dense/L1-probes-match-or-beat-SAE-probes claim is instead supported by
  Kantamneni et al., "Are Sparse Autoencoders Useful? A Case Study in
  Sparse Probing" (arXiv 2502.16681) — cite this one when presenting the
  100/768-dim sparse-probe result; frame as "consistent with this
  broader pattern, using a much simpler L1-logistic method," not as if
  sparsity itself were undiscovered. (This file previously repeated the
  now-corrected miscitation; draft/paper_draft.md §2 has the fix.)
- **Inference-Time Intervention / mass-mean shift** (Li et al. 2023, arXiv
  2306.03341) — cite as the origin of the difference-of-means steering
  method used throughout (both the original whole-residual steering and
  the new FFN-targeted patching are directly descended from this).
- **"Mechanistic Understanding and Mitigation of Non-Factual
  Hallucinations"** (arXiv 2403.18167, EMNLP Findings 2024) — cite as prior
  art for "hallucination localizes to mid-layers," so the 7-method
  layer-localization result (L8-9) is framed as confirmatory triangulation,
  not a new qualitative discovery.

## One-paragraph related-work opening (draft)

> Mechanistic accounts of LLM hallucination have so far been developed
> almost exclusively in retrieval-augmented settings. ReDeEP (Sun et al.,
> ICLR 2025) established that FFN sublayers can override retrieved context
> during RAG hallucination ("Knowledge FFN" over-retrieval), and follow-up
> work (ParamMute, NeurIPS 2025; SEReDeEP) has refined this mechanism
> without leaving the RAG regime -- both rely on benchmarks that supply
> retrieved context by construction. Whether the same FFN-dominant
> mechanism operates when there is no context to override at all -- pure
> closed-book, parametric-only generation -- has, to our knowledge, not
> been tested. Separately, a robust null result has recently been reported
> for the specific question of whether activation-space interventions
> derived from hallucination probes can *correct* a model's generated
> answer at small scale (Detection Without Correction, arXiv 2604.13068);
> we engage this directly by testing a component-targeted (FFN-sublayer,
> rather than whole-residual-stream) intervention on the same model family,
> and report the result plainly regardless of direction.
