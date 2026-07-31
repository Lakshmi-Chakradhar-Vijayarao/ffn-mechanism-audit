# Two Failure Modes That Make Interpretability Intervention Nulls Uninterpretable: A Closed-Book Case Study

**Anonymous Author(s)**
Paper under double-blind review

## Abstract

Causal-patching and flip-rate experiments in mechanistic interpretability
can produce nulls that look decisive but are not interpretable, for
reasons that are rarely checked. Using a closed-book test of whether small
models (GPT-2, Pythia-410M, Qwen2.5-0.5B) show an FFN-vs-Attention
hallucination-detection asymmetry analogous to one reported in
retrieval-augmented settings (ReDeEP), we find that the intended target
hypothesis cannot be tested on this testbed. TruthfulQA is answered
correctly by GPT-2 in only 27/817 (3.3%) of items under an
independent LLM judge -- and 0 of 283 items added specifically to
enlarge that pool were judged correct -- so any patching-based
flip-to-correct metric is floored near zero regardless of the underlying
mechanism, and a difference-of-means direction fit at the resulting
sample size fails a held-out validity check we formalize. We give that
check an exact power calculation: at the n=11 holdout this testbed
supports, the null distribution of the held-out AUROC arises from only
11{3}=165 equiprobable label arrangements and has 25
attainable values; no *observed* AUROC below 0.875 can ever be
significant at one-sided alpha=0.05; and a pure-noise direction
passes a "held-out AUROC >= 0.75" gate 13.9% of the time and a
">= 0.80" gate 6.7% of the time.

We also show that a widely-used outcome metric in this literature --
whether an intervention "corrects" a generation -- can be contaminated:
51.9% (42/81, CI [41.1,62.4]) and 53.1% (121/228, CI
[46.6,59.4]) of the nominally hallucinated completions in our two
causal-test pools -- and 53.6% (286/534) of all baseline completions
in the full labeled pool -- are degenerate repetition loops rather than
confabulations, identifiable by a single cheap, model-agnostic check that
needs no model of hallucination content. Degeneration is close to
balanced across the word-overlap (Jaccard) label's classes (55.6% of "correct"
versus 51.5% of "hallucinated" completions), so it contaminates both
arms of a flip-rate comparison rather than only one; under the validated
judge label the split is 37.0% versus 54.4%, a difference that is
not significant at this n (27 judge-correct items) but is not
symmetric either.
Removing every degenerate item does not, under the word-overlap label,
collapse the passive probe -- its AUROC survives and slightly rises, so
degeneracy-propensity is not a sufficient explanation of the passive
signal. Under the validated judge label the same subset leaves only 17
positives and the estimate is too noisy to be informative either way; we
report both.

We package these as general-purpose validity checks -- a
direction-validity gate with a formal minimum-detectable-effect
calculation, and a degeneracy pre-filter -- inside an eight-item
causal-patching validity checklist. We add a fold-seed sensitivity
result for the passive side: on these features, changing only the
cross-validation fold seed moves a component's AUROC estimate over a range
of 0.055-0.100 (SD 0.013-0.022 over 50 seeds), which is an order of
magnitude larger than the peak-versus-peak FFN-vs-Attention margins at
issue (0.003-0.011), and the argmax "peak layer" lands on three to
six different layers across twelve seeds on both 24-layer models. The
paired same-layer difference is by contrast stable, but its sign flips
between layers within an architecture.

We also report, in full, a leave-one-category-out (LOGO) diagnostic whose
interpretation changed twice under scrutiny, since the resolution is
itself the methodological point. Two permutation controls -- including one
matching pseudo-categories to real ones on both size and per-category
class composition -- show the LOGO collapse (0.62-0.66 under standard
CV to 0.48-0.49) is specific to real topic structure and *not* an
artifact of per-category averaging: random groupings recover
0.58-0.62, and the real value falls outside that null at three of the
four cells tested (the fourth, p=0.059). Decomposing the standard-CV out-of-fold scores by pair type
shows only part of the collapse comes from LOGO restricting comparisons to
within-topic pairs (within-topic AUROC 0.54-0.68; only 4.9% of
standard-CV pairs are within-topic), the rest from removing same-topic
items from training. And a ceiling calculation shows the originally
claimed mechanism -- the probe reading off per-topic correct-answer rates
-- cannot be demonstrated: topic identity alone gives AUROC 0.79
in-sample but 0.51 or below once estimated out of fold. The supported
claim is narrower than either earlier reading: *a standard random
K-fold AUROC for this probe on TruthfulQA overstates its unseen-topic
performance by roughly 0.13-0.17 AUROC, and cross-topic performance at
the layers tested is chance*, with the mechanism unidentified.

We do not claim FFN over-retrieval is present or absent in closed-book
generation. We show that this specific testbed and instrument cannot
currently answer that question, and provide the checks that would need to
pass before a claim either way could be trusted.

## 1. Introduction

A large fraction of mechanistic-interpretability claims rest on an
intervention: patch an estimated direction or activation into a model,
measure whether the model's output changes in a specified way, and read
the result as evidence about a mechanism. When such an experiment returns
a null -- the intervention changed nothing, or changed nothing
differentially between two components -- that null is usually reported as
evidence about the mechanism. This paper is about two ways that reading
can be wrong, both of which we hit directly, and neither of which is
routinely checked.

The first is a **competence ceiling**. A flip-to-correct metric can
only move if the model is capable of producing the correct answer at all.
If the base model is near-zero-accuracy on the testbed, the metric is
floored by construction, and every downstream comparison inherits that
floor. The second is an **outcome-metric contamination**. "The
generation changed from wrong to right" is only a measurement of
semantic correction if the baseline generations are wrong *answers*.
If a large share of them are degenerate repetition loops, then an
intervention that "fixes" them may only have broken the loop -- a
generic perturbation effect, not a targeted semantic one.

Both failure modes are cheap to detect and expensive to ignore. Both are
invisible in the summary statistics such papers usually report. We found
them while trying to answer a specific mechanistic question, and the
mechanistic question is now the least reliable thing in this paper; the
diagnostics are the most reliable.

**The question we set out to answer.** ReDeEP (Sun et al., ICLR 2025)
localized one mechanism by which fluency and factuality diverge in
retrieval-augmented generation: FFN sublayers overriding retrieved
context, a "Knowledge FFN" over-retrieval pattern. We asked whether an
*analogous* asymmetry -- FFN sublayer activity relating more
strongly to hallucination than Attention sublayer activity -- is
detectable in pure closed-book generation, where there is no retrieved
context to override. We stress at the outset that this is not a direct
test of ReDeEP's mechanism. ReDeEP's mechanism is defined over response
tokens with retrieved context present; every passive probe in this paper
reads the model's forward pass over the *prompt only*, before
generation (§3). These are different questions, and we describe our
results only in terms of the question we actually ran.

**What we found instead.** The intended test is not answerable on
this testbed:

-  GPT-2 answers 27/817 (3.3%) of TruthfulQA validation items
 correctly under an independent LLM judge. Enlarging the judged pool
 by 283 previously-unused items added exactly zero correct answers.
 A flip-to-correct rate on such a testbed is floored near zero, so
 the causal experiment is not underpowered in the ordinary sense --
 the outcome variable barely exists (§4.1).

-  51.9% and 53.1% of the nominally hallucinated completions in
 our two causal-test pools -- and 53.6% of all baseline
 completions in the full labeled pool -- are repetition loops, not
 confabulations (§4.2). Degeneration is near-balanced across label
 classes, so it contaminates both arms of a flip-rate comparison.

-  The difference-of-means direction that the causal test patches in
 cannot be validated. At the n=11 held-out split this testbed
 supports, the validity check's exact null is a discrete
 distribution over 25 attainable values, arising from
 11{3}=165 equiprobable label arrangements; no
 *observed* AUROC below 0.875 is ever significant at one-sided
 alpha=0.05; and a pure-noise direction passes a plausible gate
 6.7%-46.1% of the time depending on threshold (§4.3).

Given all three, the causal null we do observe (no FFN-vs-Attention
difference surviving correction at any tested configuration, under two
labels, at two injection sites, at n=467 and n=750 -- the single
nominal exception, a common-site p=0.049, does not survive Holm
correction across even its own four tests) is not evidence that the two
components are equally causal. It is what an uninformative instrument
returns.

**Contributions.**

-  **A competence-ceiling diagnostic and its consequences** (§4.1).
 We show that the testbed, not the sample size, is the binding
 constraint, by judging every item in the benchmark split rather
 than only the subset an earlier heuristic could score. This
 converts "we need more data" into "more data of this kind cannot
 help," which is a different and more actionable conclusion.

-  **A degeneracy pre-filter for flip-rate metrics** (§4.2). A
 single model-agnostic check (a 4-8 word phrase repeated 3+
 times) flags over half of this pool's "hallucinations." We report
 its rate on three pools, its balance across label classes, and its
 effect on both the causal test and the passive probe.

-  **A direction-validity gate with an exact
 minimum-detectable-effect table** (§4.3, §5). We replace "n is
 small" with the exact null distribution of a held-out AUROC over
 all n_++n_{-}{n_+} label arrangements, and tabulate,
 as a function of held-out size, the smallest detectable AUROC, the
 attainable alpha, the false-accept rate of natural gate rules,
 and power.

-  **A ten-item validity checklist** (§5) -- eight for
 causal-patching studies, two for passive probing -- distilled from
 the above plus the controls this paper's own causal section required
 (a genuinely different-source control direction, a
 common-injection-site test, correctly-read random-direction
 ensembles, equivalence testing for nulls).

-  **A fold-seed sensitivity result for passive probes** (§4.5),
 including an exact decomposition of a 0.037 AUROC discrepancy this
 repository itself contained, and evidence that the argmax "peak
 layer" is not a well-identified quantity at this sample size.

-  **A worked case study in interpreting a grouped-CV
 "collapse"** (§4.10). A leave-one-category-out diagnostic on
 TruthfulQA looked like category leakage; a subsequent review argued
 it was estimator variance; two permutation controls, a pair-type
 decomposition, and a group-variable-only ceiling calculation show it
 is neither, and leave a narrower claim -- standard K-fold CV
 overstates unseen-topic performance by 0.13-0.17 AUROC, by a
 mechanism these experiments cannot identify. We report all three
 interpretations and the checks that separate them.

We deliberately do not claim a mechanistic finding. The peak-versus-peak
FFN-vs-Attention margins this literature compares are 0.003-0.011
AUROC on our data -- three to ten times smaller than the spread a single
cross-validation fold seed produces on the same features (+/-0.03
range, ~0.037 between the two seeds this repository happened to
use, §4.5) -- and on the 24-layer models the "peak layer" itself
lands on three to six different layers across twelve fold seeds. A
same-layer paired comparison is far more stable, but its *sign*
depends on which layer is chosen. Meanwhile the causal instrument does not
clear its own validity gate. What we can support is the negative
methodological claim, and the checks that would need to pass before a
positive one could be made.

## 2. Related Work

**Parametric factual knowledge in FFN sublayers.** ROME (Meng et al.
2022) and MEMIT (Meng et al. 2023) causally locate and edit specific
facts by writing to mid-layer FFN weight matrices; Knowledge Neurons (Dai
et al. 2022) independently identifies FFN-resident neurons whose
activation correlates with specific factual recall. Our claims are far
weaker and of a different type: we do not establish an FFN-resident
mechanism, and we do not claim to have refuted one. We report that a
particular closed-book instrument cannot decide the question.

**RAG-scoped FFN over-retrieval.** ReDeEP (Sun et al., ICLR 2025)
established FFN override of retrieved context during RAG hallucination.
Follow-up work (ParamMute, NeurIPS 2025; SEReDeEP, arXiv 2505.07528;
FACTUM, arXiv 2601.05866; RAGLens, arXiv 2512.08892) refines this
mechanism or extends it with sparse-autoencoder tooling, but all remain
confirmed RAG-only. Our setting is closed-book and prompt-only (§3), so
it is at best an analogue of theirs, not a replication attempt, and a
null here says nothing about the RAG result.

**Detection without correction.** An independent finding (arXiv
2604.13068) reports that output-confidence baselines beat activation
probes above ~410M parameters, and that residual-stream steering flips
0/7 tested models' generated answers toward correct across 42
configurations on GPT-2-scale models. Our causal section reaches a
compatible null at the component level, but we are explicit that our
version of that null is uninterpretable for the reasons in §4.1-§4.3
rather than being an independent confirmation.

**Black-box detection.** Semantic entropy (Kuhn et al. 2023) and
SelfCheckGPT (Manakul et al. 2023) operate purely on output text and
generalize to closed API-only models where hidden states are
unavailable -- the actual production-relevant baseline mechanistic probes
would need to beat. We make no such comparison here and flag its absence
explicitly.

**Sparsity and steering method provenance.** The 100/768-dimension
sparse probe result (§4.9) is consistent with Kantamneni et al. ("Are
Sparse Autoencoders Useful? A Case Study in Sparse Probing," arXiv
2502.16681), which finds dense L1 probes often match or beat trained SAE
probes; it is not evidence for a claim made by the superficially similar
"A Single Direction of Truth" (arXiv 2507.23221), which does not make
this comparison. The difference-of-means steering method used throughout
descends from Inference-Time Intervention (Li et al. 2023, arXiv
2306.03341).

**Validity checklists and audits.** The closest relatives of this
paper's contribution are not mechanism papers but protocol audits: work
showing that reported detection AUROCs in this literature are frequently
artifacts of a downstream selection step or of benchmark construction
rather than of the signal claimed. Our checklist is scoped narrowly and
complementarily: it targets intervention experiments (patching an
estimated direction and reading a flip-rate), which those audits do not
cover.

## 3. Setup and Scope

**Models, data, labels.** We use GPT-2 (124M), Pythia-410M, and
Qwen2.5-0.5B-Instruct on the TruthfulQA generation validation split.
Completions are greedy-decoded (40 new tokens) from a bare
`"Q: {question} nA:"` prompt (and, for Qwen0.5B, a
chat-templated variant, §4.6). Two labels are used throughout and always
named explicitly: a *Jaccard* word-overlap heuristic against
TruthfulQA's reference answers, and a *validated* label from an
independent LLM judge (Qwen2.5-3B-Instruct). The two labels agree only
52.2% of the time on GPT-2 (kappa=0.0417, on the full 534-item
relabel); per-architecture
kappa is 0.032 (Pythia), 0.141 (Qwen0.5B-bare), and 0.084
(Qwen0.5B-chat). §6 discusses what this does and does not license.

**A disclosure that applies to every passive result in this paper.**
Every passive probe and every estimated direction reported here is
computed from the model's forward pass over the *prompt alone*,
never over any token of the model's own generated completion
(`mech-int/src/extraction/activations.py` runs one forward pass on
the prompt string; `code/02_cross_arch_component_probe.py`
hooks a single `model(**inputs)` on the same). The
correct-vs-hallucinated label is a property of a separately-generated
completion, but that completion's tokens are never passed through the
model for these measurements. So the question every passive number here
answers is: *does the model's pre-generation internal state predict
whether its own upcoming answer will be judged correct?* That is a
coherent question, and it is the one we report on. It is not the same
question as probing an already-generated hallucinated answer for features
of its content, and it is not ReDeEP's question, which is defined over
response tokens with retrieved context present. Readers should not infer
from "last-token activation" or "hidden-state probe" language that
completion tokens were involved. We state this here, before any result,
because it scopes all of them.

**Two signatures, defined in advance.** For the closed-book analogue
we set out to test, two independent signatures would jointly constitute a
positive result: (a) a *passive* signature -- FFN-derived features
detect the label with reliably higher AUROC than Attention-derived
features at the two components' respective peak layers (§4.5-§4.6); and
(b) an *active* signature -- patching an FFN-derived direction into
the FFN sublayer during generation corrects more hallucinations than an
equivalent, genuinely Attention-derived direction patched into the
Attention sublayer (§4.4). Neither is observed reliably. §4.1-§4.3
explain why, for (b), "not observed" does not mean "absent."

## 4. Results

### 4.1 The testbed's competence ceiling

Every causal-patching result in this paper measures a flip-to-correct
rate: the fraction of prompts whose baseline generation is labeled
hallucinated and whose patched generation is labeled correct. That metric
is bounded above by the model's ability to produce a correct answer at
all.

Under the validated judge label, GPT-2 is correct on 27 of 534 items
(5.1%) in the pool the original analyses used. That pool was itself
the subset of TruthfulQA's 817-item validation split that an older
Jaccard-label filter could score; 283 items had been discarded before
the judge existed. We judged all 283
(`kaggle_kernels/paper1-causal-patch-enlarged-pool/`):
**zero were judged correct**. The full-split rate is therefore
27/817 = 3.3%, exactly unchanged from the 27 correct in the original
subset.

This is a testbed-validity problem, not a sample-size problem, and the
distinction matters for what a reader should conclude. Enlarging the
causal test's prompt pool from 467 to 750 judge-hallucinated prompts
(a factor of 1.61, not "nearly double") leaves the null intact
(FFN-vs-Attention common-site p=0.18-0.84 across all four
configurations) but does not raise the flip-rate ceiling at all,
because the added prompts contributed no correct answers to flip toward.
It also does not monotonically improve resolution: the minimum detectable
odds ratio moves to 2.5-8.0 at n=750 from 3.25-6.0 at n=467
for the same common-site comparison -- tighter at three configurations
(L8/alpha=20: 3.25->2.5; L8/alpha=40: 3.25->2.625;
L9/alpha=40: 4.0->3.0) and *looser* at the fourth
(L9/alpha=20: 6.0->8.0), because that comparison's discordant-pair
count fell from 14 to 9 as the pool grew. Minimum detectable effect
depends on discordant pairs, not on n.

The general form of the check is: before running an intervention that
scores "did the output become correct," measure the base model's
unassisted correct rate on the exact evaluation set, using the same
scorer that will score the intervention. If it is near zero, the
experiment cannot distinguish mechanisms, and no amount of additional
data of the same kind will change that.

### 4.2 Degeneracy contamination of the flip-rate outcome metric

The second precondition for a flip-rate metric is that the baseline
"hallucinated" outputs are actually wrong *answers*. We checked
this with a criterion that needs no model of hallucination content at
all: flag a completion if some 4-, 5-, 6-, or 8-word phrase
occurs 3 or more times in it
(`code/04_degeneration_check.py`). The check is deterministic,
model-agnostic, and costs nothing.

On three separately constructed pools:

-  The original 70/30-split causal test set (n=81): 42/81 =
 51.9% (Wilson 95% CI [41.1%, 62.4%]) are degenerate
 repetition loops.

-  The maximum supportable scaled test set (n=228, a leaner 15/85
 split, `code/10_ffn_causal_patch_scaled.py`): 121/228 =
 53.1% (CI [46.6%, 59.4%]).

-  All 534 cached baseline completions, both label classes together
 (`code/49_nondegenerate_subset_probe.py`): 286/534 =
 53.6%; restricted to Jaccard-hallucinated completions only,
 138/268 = 51.5%.

This is a stable property of GPT-2's TruthfulQA failure mode, not a
small-sample artifact. One gap in coverage, disclosed rather than
smoothed over: the 283 prompts added in §4.1 to enlarge the judged pool
were never themselves degeneracy-checked, so the 53.6% figure covers
the original 534-item pool and the n=750 causal run inherits an
unmeasured degeneracy rate on its added prompts. Its consequence for the causal test is direct: for
roughly half the test set, an intervention that "flips" the output to a
reference-matching string is at least as consistent with breaking a
repetition loop as with correcting a hallucinated claim.

**Degeneration is near-balanced across label classes, which makes
it worse, not better.** Under the Jaccard label, 148/266 (55.6%) of
*correct*-labeled completions and 138/268 (51.5%) of
hallucinated-labeled ones are degenerate; under the validated judge
label, 10/27 (37.0%) and 276/507 (54.4%). A word-overlap
heuristic credits repetition loops as correct about as often as it
penalizes them. So degeneration does not merely add noise to one arm of a
flip-rate comparison; it is present on both sides of the label boundary
and on both sides of the intervention comparison.

**Effect on the causal test.** Restricting the n=228 scaled test
to the 107 prompts confirmed non-degenerate, the FFN-found-vs-FFN-random
null holds if anything more uniformly (McNemar p=1.000, 1.000, 1.000,
0.503 at L8/alpha=20, L8/alpha=40, L9/alpha=20,
L9/alpha=40, versus p=0.522, 0.868, 1.000, 0.659 unfiltered).
Filtering does not rescue an effect.

**Effect on the passive probe.** A live alternative explanation for
the passive probe's above-chance AUROC is that it detects
degeneracy-propensity -- "this prompt will send GPT-2 into a loop" --
rather than hallucination content. We tested this by rerunning
`code/02`'s exact probe protocol on the non-degenerate subset
only, at every layer, for both components, under both labels
(`code/49_nondegenerate_subset_probe.py`; n=248 of 534
survive the filter, 118 of them Jaccard-correct and 17 judge-correct).
Because a single fold seed moves these estimates by more than several of
the effects this paper compares (§4.5), each number below is a mean over
six CV fold seeds (0-5) rather than a single seed; per-cell values at
`code/02`'s own seed 42 are in the result JSON and do not change
any conclusion here.

**Under the Jaccard label, the passive signal survives the filter
and in fact strengthens.** FFN's peak rises from 0.6155 (L8, full pool)
to 0.6305 (L8, non-degenerate); Attention's peak rises from 0.6160
(L3) to 0.6812 (L3); and the average AUROC across all 12 layers rises
for both components (FFN 0.5619->0.6015; Attn 0.5551->0.5923). The
FFN layer-count majority is essentially unchanged (8/12->7/12).
**So the passive probe is not merely a degeneracy detector** -- if it
were, removing every degenerate completion from both classes should have
collapsed it, and instead every summary improves. Some of that
improvement plausibly reflects a cleaner label (the Jaccard heuristic's
worst errors are exactly the repetition loops it credits as correct,
§6), which is itself informative: the filter removes label noise, not
signal.

**Under the validated judge label the same test is uninformative,
and we say so rather than pick the favorable half.** The filter leaves
only 17 judge-correct items, and at that positive count the estimates
are unstable and mostly fall: FFN's peak moves from 0.6429 (L5, full)
to 0.5756 (L11, non-degenerate) and its all-layer average falls from
0.5757 to 0.4941 (below chance), while Attention's peak is roughly
unchanged (0.6772 at L5 -> 0.6813 at L5) and its all-layer average
falls from 0.6083 to 0.5442. We do not read the FFN drop as evidence
that FFN's signal specifically was degeneracy-driven: with 17 positives
split across five CV folds, neither component's estimate here is powered
to distinguish a real change from noise. What the two labels agree on is
that this test does not rescue an FFN-specific reading, and the
Jaccard-label version -- the one with enough positives to estimate at all
-- rules out the degeneracy-propensity alternative for the passive probe
as a whole.

### 4.3 A direction-validity gate, and its exact power

Before any direction is injected, does it separate the two classes on
genuinely held-out data? We split the 58-prompt training pool (18
judge-correct, 40 judge-hallucinated) into a direction-fit set (47)
and a held-out validity set (11: 3 correct, 8 hallucinated),
estimated each difference-of-means direction on the former, and measured
its scalar projection's AUROC on the latter.

At neither tested layer does either direction clear chance in the helpful
direction: L8 FFN/Attn AUROC =0.083/0.083 (bootstrap 95% CI
[0.0,0.333] both), L9 FFN =0.0 (CI [0.0,0.0]), L9 Attn =0.125 (CI
[0.0,0.5]). Three of the four CIs exclude 0.5 entirely, and an exact
Mann-Whitney test at this n_+=3, n_{-}=8 split finds them nominally
significant in the *anti-predictive* direction. All p-values here
are exact and two-sided: L8 FFN/Attn p=0.0485 each, L9 FFN p=0.0121;
L9 Attn p=0.0848 is not significant. On a consistently two-sided
convention the two L8 cells sit just under 0.05 rather than comfortably
below it (Appendix A, item 5).
Under Bonferroni across the four cells (alpha=0.0125), only L9 FFN
survives. These four p-values are recomputed and saved by
`code/51_direction_validity_mde_table.py` alongside the exact
null they come from.

**This particular anti-predictive result is a single unlucky draw.**
Redrawing which items land in the fit and holdout roles at 200 random
seeds on the identical 534-item judge-labeled pool
(`code/46_direction_validity_resplit_diagnostic.py`), the
held-out AUROC has mean 0.54-0.58 and SD 0.20-0.22, with range
[0.083,1.0] at L8 and [0.0,1.0] at L9 -- centered at or slightly
above chance. The kernel's single seed sits in the extreme low tail: only
1.5% (L8 FFN), 1.5% (L8 Attn), 0.5% (L9 FFN) and 4% (L9 Attn)
of resplits produce an AUROC at or below it. A second, independent draw
from a differently-ordered pool (the enlarged-pool run) gives
0.375/0.333 (L8 FFN/Attn) and 0.167/0.25 (L9 FFN/Attn) -- inside
the typical range. Two alternative estimators (logistic-regression
weights, Fisher LDA) fit on the identical split do no better: all 12
layer/component/estimator combinations land at or below AUROC 0.167
(`code/44_alternative_direction_estimators.py`).

**The gate cannot pass at this n, and we can say exactly how
badly.** Under the null that a direction carries no information, the
held-out AUROC is a rescaled Mann-Whitney U statistic whose
distribution is exact and enumerable over all
n_++n_{-}{n_+} label arrangements. We compute it exactly
(`code/51_direction_validity_mde_table.py`); Table~the table below
reports it as a function of held-out size at this paper's own 3{:}8
class ratio. At n=11 the null arises from only 165 equiprobable label
arrangements and the AUROC takes just 25 attainable values (spacing
1/24), so the smallest *observable* AUROC that could ever be called
significant at one-sided alpha=0.05 is 0.875 (attained exact size
0.0424). This is a statement about the observable statistic, not about
the true effect: true effects well below 0.875 can clear the threshold,
just rarely (power 0.31 at a true AUROC of 0.75, 0.72 at 0.90).
In the other direction, a pure-noise direction passes a naive "held-out
AUROC >0.5" gate 46.1% of the time, a ">= 0.75" gate 13.9%
of the time, and a ">= 0.80" gate 6.7% of the time. Reaching a
gate that both admits moderate true effects (MDE ~ 0.68) and
rejects noise at the 1% level needs roughly n=37 (10 positives).
That is nominally supplyable from this testbed's 27 correct items, but
only by spending 10 of them on the holdout and leaving 17 to fit the
direction -- so the binding constraint is not the holdout alone but the
trade-off between gate power and direction quality, which 27 positives
cannot satisfy simultaneously.

*Exact operating characteristics of a held-out direction-validity gate, as a function of held-out sample size, at this paper's own 3{:}8 positive:negative ratio. "MDE" is the smallest *observable* held-out AUROC that can be significant at one-sided alpha=0.05 under the exact null (a property of the statistic, not a bound on the true effect -- the power columns give the chance a given true effect clears it); "exact alpha" is that test's true size (discreteness makes it smaller than 0.05); "false-accept" columns give the exact probability that a pure-noise direction passes the stated gate rule; power is Monte Carlo (20{,}000 draws) under a binormal alternative. The first row is this paper's own gate.*

- n | n_+ | arrangements | MDE | exact alpha | FA >0.5 | FA >=0.75 | pow. @0.75 | pow. @0.90

- 11 | 3 | 1.65x10^{2} | 0.875 | 0.0424 | 0.461 | 0.139 | 0.31 | 0.72

- 18 | 5 | 8.57x10^{3} | 0.769 | 0.0473 | 0.500 | 0.059 | 0.51 | 0.93

- 29 | 8 | 4.29x10^{6} | 0.708 | 0.0464 | 0.490 | 0.021 | 0.69 | 0.99

- 37 | 10 | 3.48x10^{8} | 0.681 | 0.0488 | 0.493 | 0.010 | 0.79 | 1.00

- 55 | 15 | 1.19x10^{13} | 0.647 | 0.0493 | 0.496 | 0.002 | 0.92 | 1.00

- 73 | 20 | 4.30x10^{17} | 0.626 | 0.0495 | 0.498 | 0.000 | 0.97 | 1.00

- 110 | 30 | 8.37x10^{26} | 0.603 | 0.0498 | 0.499 | 0.000 | 1.00 | 1.00

The conclusion we draw is narrow and, we think, unavoidable: the
directions this paper patches were never shown to carry label-relevant
signal, and at this testbed's held-out size they could not have been.

### 4.4 The causal-patching null, and why it is not interpretable

**Design.** A difference-of-means "truthfulness direction,"
computed on FFN sublayer output over a train split only, is injected
additively into the FFN sublayer during generation, tested at L8/L9,
alpha{20,40}, against a random-direction control and against a
genuinely Attention-derived direction injected at the Attention site. The
two directions are nearly orthogonal (cosine similarity -0.054 at L8,
-0.056 at L9), so the component-specificity control is a real control
and not a relabeled copy of the treatment
(`kaggle_kernels/paper1-causal-patch-real-attn-direction/`).

**Results, under the validated label at n=467
judge-hallucinated prompts.** Flip-to-correct rates sit at 1.3-3.0%
in every one of the sixteen found/random x component x
layer x alpha conditions. FFN-found vs. Attention-found gives McNemar
p=0.167 (L8/alpha=20), 0.607 (L8/alpha=40), 1.000
(L9/alpha=20), 0.077 (L9/alpha=40) -- none significant, and
the smallest would not survive a mild multiple-comparison correction
across four configurations. Each arm is also indistinguishable from its
own random-direction control (FFN: p>=0.52 throughout; Attention:
p=0.84, 0.50, 1.00, 0.18). Rerunning the same prompts and patches
scored by the Jaccard heuristic instead gives FFN-vs-Attention p=0.439,
0.355, 0.747, 0.399 -- also uniformly non-significant.

**What this null can and cannot exclude.** Discordant pairs per
configuration are 19, 15, 9, 16. At those counts, the minimum odds
ratio an exact two-sided McNemar test can detect at p<0.05 is 3.75
(L8/alpha=20), 4.00 (L8/alpha=40), 8.00 (L9/alpha=20),
4.33 (L9/alpha=40). **This 3.75-8.00 range belongs to the
native-site FFN-found-vs-Attention-found test, whose directions are fit
on the full 58-item train pool**; it should not be conflated with the
3.25-6.00 range quoted in §4.1, which belongs to a different test --
the common-injection-site comparison from the tier-1 validated kernel,
whose directions are fit on the 47-item direction-fit subset after the
validity holdout is removed. Both are at n=467 prompts; they differ in
comparison, injection site, and direction-fit set, and their discordant
counts differ accordingly (19/15/9/16 versus 17/17/14/15). The
observed odds ratios and approximate 95% CIs (log-odds normal) are
0.46 [0.18, 1.21], 0.67 [0.24, 1.87], 1.25 [0.34, 4.66], and
3.00 [0.97, 9.30]. Only one of the four (L9/alpha=20,
[0.34,4.66]) is simultaneously consistent with FFN being several-fold
worse and several-fold better than Attention; the other three exclude one
direction or the other at the several-fold scale while still failing to
exclude a moderate effect. These are properties of the observed
discordant counts, not evidence about the two components: see "the
honest reading" at the end of this subsection for why no reading of
these intervals is licensed.

**Equivalence testing.** A TOST search finds no cell establishes
equivalence at the pre-registered OR=2.0 bound; the smallest achievable
equivalence bounds across the eight tested cells (native and common site
x four configurations) range 2.85 to 10.85
(`code/38_tier1_tost_competing_risks.py`). Both sites here are
the *tier-1* kernel's, whose directions are fit on the 47-item
subset; its native-site discordant counts are 17/13/17/18, so this is
again not the 19/15/9/16 test reported two paragraphs above, for the
same reason given there. So this test cannot
assert equivalence either.

**The random-direction ensemble, read correctly.** At the flagship
configuration (L8, alpha=40, 60 test prompts) the found FFN and
Attention directions' flip rates were compared against 20
independently-drawn random directions. **The found directions
produced 0/60 flips each; 10 of 20 random FFN directions and 8 of
20 random Attention directions also produced 0/60.** The reported
"50th percentile" (FFN) and "40th percentile" (Attention) are
therefore tie counts at the floor of the statistic, not meaningful
locations in a distribution. **This check is uninformative at this
flip-rate floor and should not be read as evidence the found direction is
special** -- in either direction. The same applies to the enlarged-pool
rerun, where the found directions land at the "95th" (FFN) and
"100th" (Attention) percentile: those are driven by 2/60 and 3/60
flip events respectively against an ensemble whose members mostly score
0/60 or 1/60. We report both runs and treat neither as evidence.

**Injection-site and dosage controls.** A common-injection-site test
(patching both directions at the shared post-block residual stream)
confirms FFN-native-site and FFN-common-site patching are mathematically
identical for GPT-2's block structure (verified byte-for-byte), and finds
the genuinely new Attention-common-site comparison non-significant at
every configuration (p=0.1435, 0.049, 0.7905, 0.6072; the nominal
0.049 does not survive Holm correction across even these four tests). A
2x3 competing-risks (flip / no-flip / degenerate) chi-square finds
one nominal p=0.0097 that likewise does not survive correction. On
dosage: measured on 100 correctly-formatted prompts, the
residual-stream norm at the injection point is 124.7 (L8) and 166.0
(L9), giving relative perturbations of 16.0%/12.0% at alpha=20
and 32.1%/24.1% at alpha=40
(`code/26_ffn_attn_dosage_diagnostic.py`). **This
quantity is equal for the FFN-site and Attention-site arms by
construction, not by measurement**: the script computes a single shared
denominator (the residual-stream norm at that layer) for both arms, so
the two arms' relative perturbation is necessarily identical. What the
diagnostic establishes is that the *correct* denominator is the
residual stream rather than each sublayer's own output norm -- which
removes the apparent 2-3x asymmetry an earlier, wrong-denominator
version of this diagnostic reported (Appendix A). It does not
independently demonstrate the absence of a dosage asymmetry, because with
this denominator no asymmetry could have been observed.

**A low-dose sweep.** At the common site with alpha{2.5,5,10}
(Jaccard label, for speed), flip rates are flat: FFN 50.00%,
45.00%, 51.67%; Attention 50.00%, 48.33%, 41.67% -- a
41.67-51.67% range overall. There is **no monotone increase**
with dose in either arm; the Attention arm is in fact monotone
*decreasing* across the three doses, the opposite of what a genuine
dose-responsive steering effect would produce.

**A differential-degeneration confound in the outcome metric
itself.** A non-trivial fraction of interventions in every condition
produce a degenerate or unparseable completion rather than a clean
correct-or-wrong answer, ranging 10.1%-46.5% across all twelve
found/random/attn-found conditions at both layers and alphas
(`results/ffn_causal_patch_scaled_results.json`). Found and
random directions are at broadly similar rates in most configurations but
diverge at L8, where the found direction degenerates *more* than a
random direction of equal norm (alpha=20: 25.9% vs. 16.7%;
alpha=40: 40.8% vs. 25.4%). This does not generalize: at L9
the ordering reverses (alpha=20: 11.0% vs. 15.4%;
alpha=40: 19.3% vs. 22.8%). We note the L8 pattern because it
is consistent with the flip-rate signal being generic perturbation rather
than targeted correction, but two of four cells run the other way and we
do not claim it as a general property. These numbers come from the
n=228, Jaccard-label scaled run
(`results/ffn_causal_patch_scaled_results.json`), not from the
n=467 validated-label run this subsection otherwise reports. Because unparseable completions are scored as
not-flipped in every condition, this differential degeneration could
mechanically penalize whichever arm degenerates more; we do not attempt a
competing-risks correction and flag it as unresolved.

**Extension beyond GPT-2, for completeness.** The same test on
Pythia-410M and Qwen2.5-0.5B-Instruct (chat-templated) is too underpowered
to support any claim. On Pythia, valid pairs collapse from n=22 at
alpha=10 (p=0.25) to n=7 at alpha=20 to n=0 at alpha=40
(generation degenerates entirely). On Qwen0.5B-chat, valid pairs are
n=2,1,0 at alpha=10,20,40 -- uninformative rather than merely
underpowered, since this instruction-tuned model's chat-style responses
rarely clear the word-overlap labeling threshold even at baseline.

**The honest reading.** Every one of these tests returns a null.
None of them is informative about component specificity, because §4.3
shows neither injected direction was ever demonstrated to carry
label-relevant content, §4.1 shows the outcome variable is floored by the
model's competence, and §4.2 shows over half the outcome events that do
occur are confounded with loop-breaking. A null from an instrument that
cannot be shown to measure anything is uninterpretable between "FFN and
Attention are equally (un)causal for hallucination" and "we injected two
vectors indistinguishable from noise, so of course nothing differed." We
report it as the latter possibility not being excluded, which is the same
as reporting the instrument as inconclusive.

### 4.5 Passive component probes: effect sizes against fold-seed noise

On GPT-2 under the Jaccard label, FFN wins 8/12 layers (two-sided
binomial p=0.39; one-sided p=0.19; neither significant). Peak FFN
layer is L8 (AUROC 0.6053); peak Attn layer is L3 (AUROC 0.6165) --
so the single best-discriminating component on GPT-2 is Attention, not
FFN. At L8, FFN direct logit attribution is higher for hallucinated
samples (5.08) than correct samples (4.85): an in-sample,
non-cross-validated, suggestive but unconfirmed "over-retrieval"
signature.

**How large is a fold seed worth?** Two numbers this repository
reports for the *same* features (GPT-2 FFN L8) differ: 0.6053 in
this section and 0.643 in the paired-delta analysis of
`code/37_paired_component_delta_auroc.py`. The cause is the
cross-validation fold seed, not the aggregation convention (mean-of-folds
versus pooled out-of-fold predictions) that might be suspected first
(Appendix A, item 7):
`code/02_cross_arch_component_probe.py` uses
`StratifiedKFold(random_state=42)`, `code/37` uses
`random_state=0`. Decomposing exactly
(`code/50_cv_seed_sensitivity_sweep.py`): mean-of-folds at seed
42 is 0.6053; mean-of-folds at seed 0 is 0.6418 (seed component
+0.0365); pooled-OOF at seed 0 is 0.6427 (aggregation component
+0.0010). **The seed accounts for 0.0365 of the 0.0374 total
gap; the aggregation convention accounts for 0.0010.**

**A 50-seed sweep, and what it does and does not undermine.** We
swept `StratifiedKFold`'s `random_state` over 50 values,
changing nothing else in `code/02`'s protocol, at two layers per
architecture and both components at each. For Pythia (L11/L4) and
Qwen0.5B (L8/L17) those two layers are that architecture's own reported
peak FFN and peak Attn layer; for GPT-2 they are L8/L9, the pair the
causal experiments patch, so GPT-2's reported Attn peak (L3) is not
itself in this 50-seed sweep (the 12-seed full-profile sweep reported
below does cover every layer). Individual
component AUROCs move substantially: GPT-2 L8 FFN spans [0.5868,
0.6422] (mean 0.6154, SD 0.0132); Qwen0.5B L17 Attn spans [0.4953,
0.5948] (mean 0.5519, SD 0.0220), and Qwen0.5B L17 FFN spans
[0.4807,0.5724] (mean 0.5219, SD 0.0174) -- straddling chance. Across all twelve
architecture/layer/component cells swept, the full range over 50 seeds
is 0.055-0.100 AUROC and the SD is 0.013-0.022. A single-seed
estimate on this data therefore carries roughly a tenth of an AUROC point
of fold-assignment spread -- an order of magnitude larger than the
0.003-0.011 peak-versus-peak margins §4.6 compares, and several times
the 0.019-0.028 margins elsewhere in this paper.

**The paired same-layer difference, by contrast, is stable, and we
report that rather than overstating the seed result.** Because both
components are fit on the same folds, their difference cancels most of
the fold-assignment noise: at GPT-2 L8, Delta=+0.0688+/-0.0155 with
FFN ahead in 50/50 seeds; at Pythia L11, +0.0512+/-0.0209, FFN ahead
in 98%; at Qwen0.5B L8, +0.0118+/-0.0204, FFN ahead in 70%. The
sign is layer-dependent, not architecture-dependent: at Pythia L4,
Delta=-0.0569+/-0.0212 with FFN ahead in 0% of seeds, and at
Qwen0.5B L17, -0.0300+/-0.0231, FFN ahead in 10%. **So the
FFN-vs-Attention answer is determined by which layer is compared, and the
peak-versus-peak comparison this literature (and this paper's earlier
draft) relies on is precisely the fragile way to choose it**, because
argmax over near-tied layers is the statistic most exposed to fold-seed
noise.

**A paired test of the actual estimand.** The peak-versus-peak
comparisons above are visual: two separately-estimated CV means checked
against their own fold SDs, though the FFN and Attention probes are fit on
the same samples and folds. The paper's real estimand is their difference,
Delta=AUROC_{FFN}-AUROC_{Attn}. Using
already-cached raw per-sample features (Pythia, Qwen0.5B) and GPT-2's
vendored mech-int activations (no new model inference), we compute
out-of-fold predicted probabilities for both components from the same
folds and a BCa bootstrap 95% CI on Delta over 2000 resamples
(`code/37`). At each architecture's own FFN-peak layer under this
aggregation: GPT-2 (L8) Delta=+0.067, CI [+0.012,+0.122]; Pythia
(L11) Delta=+0.047, CI [-0.002,+0.100]; Qwen0.5B (L20)
Delta=+0.053, CI [-0.007,+0.113]. At each architecture's own
Attn-peak layer the sign reverses as expected: GPT-2 (L3)
Delta=-0.085, CI [-0.137,-0.036]; Pythia (L4) Delta=-0.113, CI
[-0.162,-0.064]; Qwen0.5B (L8) Delta=-0.032, CI [-0.084,+0.023].

**These do not all agree with the 50-seed sweep above, and the
disagreement is the same effect again.** Two cells are measured by both
procedures. They agree at GPT-2 L8 (+0.067 versus +0.0688) and Pythia
L11 (+0.047 versus +0.0512), and they disagree at the other two:
Pythia L4 gives -0.113 here against a sweep mean of -0.0569, and
Qwen0.5B L8 gives -0.032 here against +0.0118 -- a sign flip. The
sweep is a mean over 50 fold seeds at `code/02`'s aggregation;
`code/37` is one seed at pooled-OOF aggregation. So the "stable"
claim we make for the paired difference is a within-protocol claim: at a
fixed aggregation, Delta varies little across fold seeds (SD
0.016-0.023). It is not a claim that two different protocols agree on
Delta at a given layer, and at two of four shared cells they do not.

Averaged across *all* layers rather than only the peaks -- the
summary least vulnerable to selection bias -- Delta is small on every
architecture (GPT-2 +0.0017+/-0.036; Pythia +0.0005+/-0.042;
Qwen0.5B -0.0021+/-0.034), and a layer-weighted pooled estimate across
all three architectures gives Delta=-0.0003 with between-architecture
variance of 3.9x10^{-6}. That is the properly pooled null.

**Peak layers are not stable.** `code/37`'s pooled-OOF argmax
puts Qwen0.5B's FFN peak at L20 and its Attn peak at L8, the reverse of
the mean-of-folds peaks reported elsewhere in this paper (FFN L8, Attn
L17). This is not the two aggregation conventions disagreeing about which
layer wins (Appendix A, item 7). The aggregation convention is worth
0.001 AUROC and the fold seed is worth 0.037 on the same data; the two
analyses also differ in fold seed, and argmax over 24 near-tied layers
is exactly the statistic most sensitive to it.

We measured this directly by recomputing the full per-layer AUROC profile
at 12 different fold seeds and recording the argmax
(`code/50`). On Qwen0.5B, the FFN peak lands on **four**
different layers across 12 seeds (L12 in 6, L8 in 3, L20 in 2, L2
in 1) and the Attention peak on **six** (L5 in 6, L17 in 2, and
L7, L8, L12 and L23 once each). Both of the "colliding" assignments -- FFN L8 / Attn L17
from `code/02` and FFN L20 / Attn L8 from `code/37` -- are
draws from that distribution. Pythia is similar (FFN peak on 3 distinct
layers, modal L11 in 10 of 12 seeds; Attention peak on 5, with no
modal layer at all -- L4 and L19 tie at 4 of 12 seeds each, then L2 in
2, L9 and L17 once each). GPT-2, with half as many layers, is the stable case for FFN
(L8 in 12/12 seeds) but not for Attention, which also ties (L3 in 6,
L6 in 6).
**There is no estimator disagreement to explain: "the peak layer"
is not a well-identified quantity for either component on the 24-layer
models at this sample size.** Any claim that
depends on a specific peak layer -- including the peak-depth-fraction
comparison in §4.6 -- inherits that instability.

We also checked "peak AUROC" itself for winner's-curse selection bias
using nested cross-validation (select the peak layer on inner folds only,
evaluate on a held-out outer fold). Nested-CV peaks are uniformly lower
than naive argmax peaks -- GPT-2 FFN 0.643->0.581, GPT-2 Attn
0.632->0.600; Pythia FFN 0.632->0.584, Pythia Attn
0.666->0.582; Qwen0.5B FFN 0.563->0.521, Qwen0.5B Attn
0.570->0.515. On GPT-2 the correction flips which component leads
(naive: FFN 0.643 vs. Attn 0.632; nested: Attn 0.600 vs. FFN
0.581). We do not read this as Attention "really" winning -- the
corrected margin is itself within noise -- but as a concrete
demonstration that peak-AUROC comparisons can select the noisier rather
than the truer component.

One consequence should be stated rather than left for a reader to notice:
the "naive argmax" column above disagrees with §4.6's peak table on
*which component wins, on all three architectures*. Here FFN leads
on GPT-2 (0.643 vs. 0.632) and Attention leads on Pythia (0.666
vs. 0.632) and Qwen0.5B (0.570 vs. 0.563); §4.6, computing the
same quantity at `code/02`'s fold seed and aggregation, has exactly
the opposite winner in each case. Both sets of numbers are correct for
their own protocol. Three sign flips out of three, produced by nothing
but a fold seed and an aggregation convention, is the strongest single
piece of evidence in this paper that the peak-versus-peak
FFN-vs-Attention comparison is not measuring anything at this sample
size, and we report §4.6's version only because it is the protocol the
rest of the passive results use -- not because it is the right answer.

### 4.6 Cross-architecture passive results

[Real Kaggle data, N=605 Pythia / N=513 Qwen0.5B; per-layer numbers in
`results/cross_arch_component_probe_*.json`.] Under the Jaccard label and a
bare prompt template, FFN wins a numerical majority of layers on all
three architectures (66.7%, 66.7%, 58.3%); per-architecture
two-sided p-values are 0.39/0.15/0.54, one-sided 0.19/0.076/0.27 --
all non-significant. Pooled across all 60 layers, 38/60 FFN wins
gives a nominally significant one-sided p=0.026, but this is not a
valid inferential instrument: 60 layers within only 3 architectures
are strongly autocorrelated, not independent trials. The only cleanly
poolable count is architecture-level (3/3), directionally consistent but
far too small an n to test formally.

On peak AUROC, Attention is the single best-discriminating component on
**one** of three architectures (GPT-2). Pythia's peak favors FFN
(L11 =0.6181 vs. Attn L4 =0.6115); Qwen0.5B's also favors FFN (L8
=0.5657 vs. Attn L17 =0.5625). Every one of these margins is a small
fraction of the relevant peak's own cross-validation SD: GPT-2's 0.011
margin against SDs of 0.0557/0.0427 (about a quarter to a fifth),
Pythia's 0.0066 against 0.0442/0.0345 (about a seventh to a fifth),
and Qwen0.5B-bare's 0.0032 against 0.0628/0.0423 (about a twentieth
to a thirteenth) -- and, per §4.5, all of them are a small fraction of
what changing the fold seed alone is worth.
**The uniform statement is: the peak-component question is within
measurement noise on all three architectures tested.**

**A template confound on Qwen0.5B.** Qwen2.5-0.5B (~494M
parameters, the largest model tested) was queried with a bare
`Q: ... A:` template rather than its chat template -- genuinely
out-of-distribution usage for an instruction-tuned model. Rerunning with
the proper chat template (`code/02_cross_arch_component_probe.py qwen05chat`; only prompt construction changed) reverses the
peak-component result: Attention becomes the peak (L4 =0.5988+/-0.0438)
over FFN (L4 =0.5704+/-0.0186), and the layer-count result reverses too
(FFN wins 11/24, 45.8%, versus 58.3% before). The corrected
margin (0.0284) is still smaller than Attention's own CV SD
(0.0438), so "Attention now clearly wins" would overclaim in the same
way "FFN wins 3/3" did. The finding is the direction of the flip, not a
newly-resolved winner. Peak-FFN depth fraction also shifts, from 33.3%
(L8/24, bare) to 16.7% (L4/24, chat).

**Re-probing under the validated label.** We relabeled every
completion on all three architectures with the LLM judge and reran the
component probe under both labels
(`code/24_llm_judge_score_all_architectures.py`,
`results/llm_judge_relabel_summary.json`), and separately
extended the check to GPT-2 itself
(`code/29_gpt2_full_validated_relabel_rerun.py`, all 534
samples). Absolute AUROCs rise substantially on every architecture (GPT-2
0.605/0.616->0.698/0.717 FFN/Attn peak; Pythia
0.6181/0.6115->0.7353/0.7494; Qwen0.5B-bare
0.5657/0.5625->0.7127/0.6992; Qwen0.5B-chat
0.5704/0.5988->0.6603/0.6412), and FFN's numerical majority is
restored on Qwen0.5B-chat (11/24 under Jaccard to 18/24 under the
validated label) -- so the template-reversal finding is itself
label-sensitive. Stated completely: under the validated label FFN's
majority *rises* on three of four architecture/template conditions
(Pythia 16/24->18/24; Qwen0.5B-bare 14/24->20/24; Qwen0.5B-chat
11/24->18/24) and only GPT-2 moves the other way, reversing further
toward Attention (8/12->2/12). FFN holds a layer-count majority on 3
of 4 conditions under either label -- with different members (GPT-2 flips
out as Qwen0.5B-chat flips in). Which component *peaks* is less
stable still: only 2 of 4 conditions keep the same peak winner under both
labels (GPT-2: Attention both times; Qwen0.5B-bare: FFN both times), and
the other two flip in opposite directions.

**A class-imbalance caveat on every validated-label number above.**
The judge label is severely imbalanced toward "hallucinated": GPT-2
27/534 correct (5.1%), Pythia 29/605 (4.8%), Qwen0.5B-bare
63/513 (12.3%), Qwen0.5B-chat 73/433 (16.9%). Cross-validated
AUROC at this imbalance is substantially noisier: on GPT-2 the FFN/Attn
peak AUROC standard deviations roughly double under the validated label
(FFN 0.0557->0.1115; Attn 0.0427->0.1253), so GPT-2's own
0.698/0.717 margin (0.019) is well within one SD of either peak.
The rising absolute AUROCs and the layer-majority shifts are real
properties of the re-analysis, not coding errors, but they are suggestive
of a real, under-characterized effect of label quality on this probe --
not tighter estimates than the numbers they revise.

{figures/ffn-attn-comparison.pdf}

*Peak AUROC for FFN vs. Attention across every tested condition, with error bars showing each peak's own cross-validation standard deviation. The margin between components is within one CV~SD of overlap in all four conditions, and Qwen0.5B -- whose narrow bare-template FFN edge (0.0032, the smallest of the three) is the one that reverses -- appears under both templates, favoring Attention once queried with its proper chat template. §4.5 shows that changing only the cross-validation fold seed moves these estimates by more than the margins shown.*

### 4.7 Two further causal instruments

**ROME-style causal tracing.** Additive mean-shift steering is a
comparatively weak causal instrument. We replace it with causal tracing
(Meng et al. 2022), adapted to closed-book QA (corrupting the whole
question span, since no single clean "subject span" applies): a clean
run scores a forced-choice logit_diff between the correct and
incorrect reference answer's first token; a corrupted run adds Gaussian
noise to the question-span embeddings; a restoration sweep patches each
(layer, component) one at a time; a specificity control repeats this with
a mismatched example's activation
(`code/08_rome_style_causal_tracing.py`).

At the maximum powered sample (n_{valid}=67, pre-registering the
joint 24-test correction as primary), FFN shows no specific restoration
effect anywhere. Attention's smallest-p candidate (L9, own-shuffled
=+0.151; attn L7's effect is larger at +0.163 but its
p=0.0146 is not) does not survive the joint Holm-Bonferroni threshold
(p=0.012 vs. 0.05/24=0.00208) -- attenuated from an earlier,
lower-powered n=45 pass where the same cell was larger (+0.214,
p=0.0026) under a less conservative per-family scoping. Instead the FFN
site at L9 (`mlp_9` in the tracing artifact, the same sublayer
called FFN throughout this paper) gives an *anti-specific* result, where a mismatched example's
activation restores discrimination *better* than the example's own
which clears the strict joint threshold (p=0.00086, own-shuffled
=-0.203). That is not in tension with "no specific restoration effect
anywhere": the surviving cell runs the wrong way. We do not claim the anti-specific MLP L9 result is a stable
finding rather than one more stopping point in a noisy series. Neither
Pythia-410M nor Qwen2.5-0.5B shows any layer or component surviving
correction under either framing (Pythia's smallest uncorrected
p=0.0645, Attn L20; Qwen's p=0.0041, Attn L15).

Rerun under the validated judge label
(`code/40_rome_style_causal_tracing_validated.py`), averaging
10 independent corruption draws per example and ensembling the
mismatched-donor control over 10 random donors, the judge-correct pool
is small (17 usable candidates out of 27 total judge-correct GPT-2
samples, after requiring a parseable correct/incorrect answer pair) --
the same competence ceiling as §4.1, reappearing in a different
instrument. Across individual corruption draws, only 53.5% show real
degradation from corruption, underscoring why averaging over draws rather
than selecting on one is the right correction. No cell survives
Holm-Bonferroni at either this test's own 24-comparison family or a
stricter 120-test family across all three architectures; the two
smallest uncorrected p-values are both Attention layers (L10 p=0.026,
L11 p=0.023). One limitation is disclosed rather than fixed: the
forced-choice score remains first-token log-odds, not a length-normalized
full-sequence log-probability.

**A sparse-feature intervention.** A dense mean-difference direction
averages over every latent factor that differs between conditions; if a
correction signal exists but is carried by a small number of sparse
features, additive steering along the dense direction would dilute it. We
substitute `jbloom/GPT2-Small-SAEs-Reformatted`'s layer-8 SAE
(d_{sae}=24{,}576, trained on 300M tokens of OpenWebText) -- a
disclosed mismatch on two axes at once: wrong hookpoint (residual stream,
not FFN-sublayer output) and wrong training distribution. **0 of
24,576 features survive Benjamini-Hochberg FDR at q=0.05 on this
paper's own 534-example dataset**
(`results/sae_feature_clamp_paper1.json`), so the causal clamp
step was never reached. This is a null one stage earlier than §4.4's
test, but bounded by instrument mismatch: either axis alone could explain
zero surviving features. Running the identical procedure on a companion
dataset (HaluEval, n=500, same SAE, same layer) as a positive control
(`code/15_sae_feature_gating_utility.py`) finds 331/24{,}576
features surviving FDR (best p=4.8x10^{-11}), confirming the null
above is not a feature-selection bug. Even there, the causal clamp shows
no specificity at any strength (p=1.000, 0.508, 1.000 at
eta=10,20,40;
`results/sae_feature_clamp_combined.json`).

### 4.8 Difficulty controls

The ~0.03 AUROC margin over a surface-feature baseline (0.605
vs. 0.576) is small enough that a generalized question-difficulty
signal remains a live alternative to a hallucination-specific one. We
test this two ways on GPT-2, matching correct/hallucinated groups within
10 quantile bins of (1) mean token-level output entropy and (2) the
out-of-fold predicted probability of a logistic regression over all 6
surface features, then re-probing the identical FFN/Attn comparison on
each matched set (`code/06_difficulty_matched_control.py`).
Entropy-matching retains n=492/534; the composite match retains
n=462/534. Neither proxy correlates significantly with correctness
before matching (entropy r=0.045, p=0.295; composite r=0.024,
p=0.578) -- this dataset never had a statistically detectable
difficulty confound for either proxy to remove, so this is survival under
a weak test, not dissociation under a strong one.

The signal survives both matches essentially unchanged (entropy-match:
FFN 0.6085+/-0.0442, Attn 0.6102+/-0.0285; composite-match: FFN
0.6255+/-0.0510, Attn 0.6253+/-0.0926; vs. 0.6053+/-0.0557 /
0.6165+/-0.0427 unmatched), with a label-permutation test (500
shuffles) giving both components p=0.0020 -- the permutation floor at
this shuffle count. Extending to the other architectures
(`code/11_multi_arch_difficulty_matched_control.py`) splits by
architecture: Pythia replicates GPT-2's survival (FFN 0.6327+/-0.0513,
Attn 0.6093+/-0.0221, both p=0.0020, 89.3% retention), Qwen0.5B
does not (FFN 0.5277+/-0.0434, p=0.230; Attn 0.5282+/-0.0320,
p=0.206, 88.3% retention).

A gradient-reversal probe attaches a second head predicting the entropy
proxy from the same shared representation, back-propagating its
*negative* gradient
(`code/17_gradient_reversal_adversarial_probe.py`). The
adversarial pressure works: held-out R^2 for the entropy head is
negative for both components (-0.71 FFN, -1.44 Attn). Hallucination
AUROC nonetheless survives above chance for both (FFN 0.5944+/-0.0624,
Attn 0.6134+/-0.0542, both p=0.0050 against a 200-shuffle floor).

Rerunning the matched controls under the validated label
(`code/30_difficulty_matched_control_judge_label.py`) is
inconclusive: the severe class imbalance leaves only 54/534 samples
(10.1%) after matching, 27 per class, roughly 5 per CV fold, and
the two proxies then disagree (entropy-only: FFN 0.7840, p=0.0060;
Attn 0.7187, p=0.0200; composite: FFN 0.3987, p=0.82; Attn
0.3680, p=0.89). We read neither as informative. This control also
probes the Jaccard-label peak layers (FFN L8, Attn L3) rather than the
validated label's own shifted peaks -- a layer mismatch we did not
correct for. The difficulty-dissociation result should therefore be read
as established only under the Jaccard label.

### 4.9 Layer localization, and its label-dependence

For context on where the probes above are read, we report GPT-2 layer
localization from the vendored mech-int artifacts (verified byte-identical
to the originating project's logs;
`code/00_verify_vendored_mechint_numbers.py` reprints each).
Under the Jaccard label, six methods converge on layers 8-9: dense probe
peak L9 (0.5827); sparse L1 probe peak L9 (100/768 active dims, 87%
sparse, CV AUROC 0.589, not the inflated in-sample 0.874);
token-position probe peak L8 last-token (0.6036); component probe FFN
peak L8 (0.6053) vs. Attn peak L3 (0.6165); steering peak improvement
at L9; gold-token logit-lens divergence at L8. A seventh method, DLA
magnitude, does not join: L10 (+0.90) and L11 (+0.71) are larger than
L9.

Three of the six are weaker than "converges" implies. The dense probe's
L9 peak is a near-tie: 0.5827275 at L9 against 0.5826826 at L12, a
margin of 4.5x10^{-5} AUROC, about a thousandth of that layer's own
cross-validation standard deviation (0.0497), with L10 (0.5805) and
L11 (0.5791) also within 0.004 -- so "dense probe peaks at L9" is an
argmax over noise in the same sense as the steering result below.
Steering's "peak
improvement" at L9 is 0.0015 AUROC points, and it is the argmax over a
52-cell grid (13 layers x 4 steering strengths) in which the
improvement is exactly zero at 11 of the 13 layers; the only other
non-zero cell is +0.0001 at L10. Against a between-layer spread of
0.520-0.609 in the unsteered baseline AUROC itself, an argmax over a
statistic that is identically zero almost everywhere does not localize
anything. (The sweep artifact records no per-layer cross-validation
standard deviations, so we do not report a noise band for it.) The logit-lens analysis computes
two divergence-layer estimates from the same run: the plain
correct-vs-hallucinated divergence peaks at layer 1; only the gold-token
variant peaks at L8.

**This convergence does not survive relabeling.** Rerunning the four
methods that consume only cached activations under the full-534 judge
label (`code/29_gpt2_full_validated_relabel_rerun.py`, same
probing code, only the label array swapped), the peak layer moves: dense
probe L7 (0.6444), sparse L1 probe L7 (now 19/768 active dims, 98%
sparse, CV AUROC 0.6260), token-position probe L6 (0.6961), DLA's
largest FFN difference from L10 (+0.90) to L11 (+1.81). L8-9 is a
property of the Jaccard label's noise pattern on this dataset, not a
label-independent localization. Read together with the class-imbalance
caveat in §4.6 and the fold-seed sensitivity in §4.5, the specific peak
layers under either label should be treated as unstable, not as competing
precise estimates.

### 4.10 A leave-one-category-out result whose interpretation changed twice

We include this subsection in full, including the interpretations we
abandoned, because it is the most directly useful part of this paper for
a reader running similar diagnostics. An earlier draft reported its first
half as a headline finding; an independent review then argued the whole
thing was an estimator artifact; the controls reported here support
neither reading exactly.

**The diagnostic, and its first reading.** TruthfulQA questions
cluster into 38 topical categories, and standard random K-fold CV --
the protocol every passive probe number in this paper uses -- can place
same-category questions in both a fold's train and test split. Category is
not independent of the label: on this 534-item GPT-2 pool, among the
10 most frequent categories (each n>=15) the judge-correct rate
ranges from 0% (Fiction, Paranormal, Stereotypes) to 10.5%
(History, Conspiracies), and across all 38 from 0% to 28.6%
(Confusion: People, n=7). We ran a leave-one-category-out (LOGO) CV
re-test for the component probe at GPT-2 L8/L9
(`code/47_category_leakage_diagnostic.py`): train on 37
categories, test on the held-out one, repeated for every category with
both classes present (16 of 38 qualify; the other 22 have zero
correct-labeled items). Standard 5-fold CV gives AUROC 0.622 (L8 FFN),
0.632 (L8 Attn), 0.616 (L9 FFN), 0.663 (L9 Attn); LOGO-CV gives
0.479, 0.491, 0.482, 0.489 (SD 0.27-0.35 across folds). The
first reading was: category-clustering leakage explains the probe's
signal.

**The objection: is the "collapse" just the LOGO estimator's own
variance?** The objection has real force. The LOGO implementation averages
*per-category* AUROCs, which is a different estimand from the
per-fold average over random folds that standard CV reports. With 22 of
38 categories having zero positives and the surviving ~16
averaging ~1.7 positives each, each fold's AUROC is close to a
coin flip. A naive standard-CV-versus-LOGO comparison is also not
significant on its own terms (Welch p=0.09-0.16 across the four
cells) -- but that comparison is between two different estimands, so its
p-value does not answer the question either.

**Two permutation controls settle whether the estimator is to
blame.** We rerun the identical LOGO procedure
(`code/47::probe_leave_one_category_out`, reproduced
unmodified) on group assignments that are random by construction, 100
draws each (`code/48_permuted_pseudocategory_control.py`):
*size-matched*, permuting the category-assignment vector so group
sizes are preserved exactly; and *size- and class-matched*, permuting
positives among positive slots and negatives among negative slots so that
every pseudo-category has exactly the same n *and* the same number
of correct-labeled items as the real category it replaces -- which makes
the usable-fold count identical to the real one (16, versus 17.4 on
average under size-matching alone), removing the alternative explanation
that the 16 usable real categories are simply an unusual subset.

**Random groupings do not reproduce the collapse.** Under both
controls, LOGO on pseudo-categories recovers approximately the standard-CV
AUROC rather than chance:

*Leave-one-category-out AUROC on real TruthfulQA categories versus two permuted-pseudo-category nulls (100 draws each), GPT-2, judge label. Empirical two-sided p is computed against the corresponding null; 0.020 is the attainable floor at 100 permutations.*

- Cell | standard 5-fold CV | real LOGO | size-matched null (p) | size+class-matched null (p)

- L8 FFN | 0.6215 | 0.4788 | 0.6160+/-0.0405 (0.020) | 0.6152+/-0.0407 (0.020)

- L8 Attn | 0.6318 | 0.4907 | 0.5869+/-0.0461 (0.079) | 0.5812+/-0.0423 (0.059)

- L9 FFN | 0.6157 | 0.4816 | 0.6135+/-0.0355 (0.020) | 0.6107+/-0.0411 (0.020)

- L9 Attn | 0.6632 | 0.4891 | 0.6160+/-0.0418 (0.020) | 0.6180+/-0.0413 (0.040)

The per-category-averaging estimand is therefore *not* biased
downward at this sample size: on random groupings of identical size and
class composition it returns 0.58-0.62, within noise of the
standard-CV values. The real-category LOGO value sits below that null at
every cell, and outside it at three of four (0, 0, and 1 of 100
size- and class-matched draws fall at or below the real value at L8 FFN,
L9 FFN, and L9 Attn; L8 Attn is the exception at 2/100, empirical
p=0.059). **The collapse is specific to real topic structure, not
an artifact of the estimator.** We report this having expected, and
initially predicted, the opposite outcome.

**A second thing LOGO changes, which has to be separated out.**
Leave-one-category-out alters the protocol in two ways at once, and only
one of them is about leakage. (i) It changes which pairs enter the AUROC:
a per-category AUROC compares a positive and a negative drawn from the
*same* category, whereas standard K-fold CV pools all pairs, of
which only 4.9% (675 of 13{,}689) are within-category on this pool.
(ii) It removes same-category items from the training split. We separate
these by decomposing the *standard* CV's own out-of-fold scores into
within- and between-category pairs (`code/48`); this holds the
training protocol fixed and varies only the pair set:

*Where the standard-CV AUROC's discrimination lives. Computed on the identical out-of-fold predicted probabilities from standard 5-fold CV, split by whether the (positive, negative) pair comes from the same TruthfulQA category. "LOGO" repeats the within-category restriction *and* removes same-category items from training.*

- Cell | pooled OOF | between-topic | within-topic | LOGO

-  | (13,689 pairs) | (13,014 pairs) | (675 pairs) | (within, disjoint train)

- L8 FFN | 0.6152 | 0.6193 | 0.5363 | 0.4788

- L8 Attn | 0.6153 | 0.6163 | 0.5970 | 0.4907

- L9 FFN | 0.6067 | 0.6094 | 0.5541 | 0.4816

- L9 Attn | 0.6494 | 0.6475 | 0.6844 | 0.4891

Pair composition explains part of the gap but not most of it, and at one
cell none of it: within-topic AUROC on the same scores is 0.536-0.684,
and at L9 Attn it is *higher* than the between-topic value. The
remaining drop -- from 0.54-0.68 with same-topic items in training to
0.48-0.49 without them -- is the part attributable to training-set
topic overlap, i.e. to what "category leakage" should have meant. It is
substantial (0.06 to 0.20 AUROC depending on cell, largest at L9
Attn) and, per the permutation controls above, not estimator noise.

**But we cannot name the mechanism, and the original naming was
wrong.** The originally claimed mechanism -- the probe reading off each
topic's correct-answer rate -- makes a checkable prediction: a classifier
given *only* the category label should recover much of the
0.62-0.66 AUROC. Computed on the identical pool and labels
(`code/48`), an in-sample per-category correct-rate rule reaches
AUROC 0.7938, but leave-one-out gives 0.5054 and this paper's own
5-fold CV protocol gives 0.4863 (pooled-OOF) and 0.4789
(mean-of-folds). **Estimated the way any real probe would have to
estimate it, topic identity alone is at or below chance.** We are careful
about what this does and does not license: the in-sample value of 0.7938
shows that *if* per-category rates were known exactly they would
suffice to exceed the probe's own AUROC, so a topic-base-rate account
cannot be excluded on effect-size grounds; what it shows is that with 27
positives spread over 38 categories those rates cannot be estimated
accurately enough for that account to be tested directly here. The
competing accounts are not separated by any experiment we ran, and we do
not claim to distinguish them. They include: per-topic base rates
themselves; *continuous* topic-correlated features, which a
768-dimensional probe fit on ~427 items can exploit far more
efficiently than a 38-level lookup table estimated from ~1.7
positives per level, and which is the account the ceiling calculation is
*least* able to exclude; within-topic near-duplicate and paraphrase
structure; and topic-specific feature geometry that simply does not
transfer.

**What survives, stated narrowly.** Both readings we started from
overreach. What the evidence supports is: *a standard random
K-fold AUROC for this probe on TruthfulQA overstates its performance on
an unseen topic, by roughly 0.13-0.17 AUROC, and cross-topic
performance at the two layers tested is at chance.* The drop is specific
to real topic structure rather than to the estimator (permutation
controls), is only partly explained by the change in which pairs are
compared (Table~the table below), and is not attributable to any single
mechanism we can identify.

**Consequences for this paper's other numbers.** Every passive AUROC
in §4.5, §4.6 and §4.9 should be read as a *within-distribution*
number that includes cross-topic comparisons -- not as an estimate of what
the probe would do on a new topic. This does not differentially favor
either component (FFN and Attention drop by 0.14, 0.14, 0.13 and
0.17 AUROC), so it does not change the FFN-vs-Attention comparison, and
it does not affect any causal result in §4.3-§4.4, which use no K-fold
CV. It does mean the passive signature defined in §3 was never
demonstrated in a form that would transfer to a new topic.

**Scope of this check, and what we removed.** It covers GPT-2 only,
at L8/L9, for the component probe, under the judge label, with 16
usable folds. We did not run it for the other layer-localization methods
in §4.9. An earlier draft extended the same LOGO check to Pythia and
Qwen0.5B and reported a "heterogeneous," architecture-dependent leakage
result; **we have removed that analysis rather than repair it**,
because it was confounded independently of anything above: the GPT-2
diagnostic used last-token extraction, judge labels, and a 5.1%
positive rate, while the cross-architecture kernel used mean-pooled
extraction, Jaccard labels, and a ~47% positive rate -- three
simultaneous protocol differences fully aliased with architecture, so no
architecture-specific conclusion was identifiable from it. The files are
retained in the repository for the audit trail and are not cited as
evidence.

**The transferable lesson.** A grouped-CV "collapse" is ambiguous
between three quite different things -- group leakage, a change of
estimand, and failure to generalize across groups -- and the ambiguity is
resolvable cheaply. Run all three checks: (a) a permuted-group control,
size- and class-matched, which separates estimator behavior from real
group structure; (b) a group-variable-only ceiling under the same CV
protocol, which tests whether the group variable could carry the claimed
effect at all; and (c) an explicit statement of which estimand each
protocol computes. We ran none of the three before first making the
claim, and each of the three changed our conclusion.

## 5. The Causal-Patching Validity Checklist (and Two Checks for Passive Probing)

The constructive contribution of this paper is a checklist, exportable
independent of anything about FFNs. It applies to any study that
estimates a direction (or feature, or activation) from labeled data,
patches it into a model, measures an outcome rate, and draws a causal
conclusion. Each item below is one this paper's own flagship experiment
either failed or needed in order to be read correctly; the section
references point to where.

-  **Is the outcome metric floored by the base model's
 competence?** Measure the unassisted correct rate on the exact
 evaluation set with the exact scorer that will score the
 intervention, and report it. If it is near zero, a flip-to-correct
 rate cannot separate mechanisms, and more data of the same kind will
 not help (§4.1: 27/817 = 3.3%; 0 of 283 added items correct).

-  **Are the baseline "failures" the failure mode you think they
 are?** Run a degeneracy pre-filter before trusting a flip-rate.
 Ours is a single deterministic check -- a 4-8 word phrase
 repeated 3+ times -- and flags 51.9% and 53.1% of the
 nominally hallucinated completions in our two causal-test pools, and
 53.6% of all baseline completions in the full labeled pool,
 near-balanced across label classes (§4.2). Report the rate on both arms and on both label
 classes; a filter that removes items from only one arm creates its
 own confound.

-  **Is the estimated direction validated on held-out data before
 any intervention, with a stated minimum detectable effect?** Do not
 report "n is small." The exact null distribution of a held-out
 AUROC is enumerable over all n_++n_{-}{n_+} label
 arrangements; report the resulting MDE, the attainable alpha, and
 the false-accept rate of the gate rule actually used
 (Table~the table below). At this paper's own n=11 the gate cannot
 detect any true AUROC below 0.875, and a pure-noise direction
 passes plausible gates 6.7%-46.1% of the time. A patching
 result whose direction has not cleared such a gate is uninformative
 about the mechanism in either direction (§4.3).

-  **Is the "control" direction genuinely from a different
 source?** A control that is the treatment direction relabeled, or
 injected at a different site, cannot establish component
 specificity. This paper's earlier version made exactly that mistake
 (Appendix A); the corrected control is a difference-of-means
 direction estimated from the other component's own activations, and
 is verified near-orthogonal to the treatment (cosine -0.054/-0.056)
 before use (§4.4).

-  **Does the injection site confound "which representation"
 with "where in the computation"?** Patching two different sublayers
 at different points in a block entangles component identity with
 injection site. A common-site test disentangles them -- and can
 reveal that two apparently different interventions are
 mathematically identical, as FFN-site and common-site patching are
 for GPT-2's block structure (verified by direct algebra and a
 byte-for-byte empirical check, §4.4). Relatedly, if a dosage
 diagnostic normalizes both arms by a single shared denominator, its
 finding of "equal dosage" is true by construction and should be
 labeled as such (§4.4).

-  **Is a null distinguishable from a genuinely random
 direction -- and is the ensemble check even informative at the
 observed rate?** A found direction whose effect sits inside the
 empirical distribution of random-direction effects has not been
 shown to do anything a random direction would not. But read the
 percentile correctly: when the found direction scores 0/60 and
 half the random ensemble also scores 0/60, the resulting "50th
 percentile" is a tie count at the floor and carries no information
 (§4.4). Report the raw counts alongside any percentile.

-  **Is a null reported at the resolution the claim needs, and is
 equivalence tested rather than assumed?** Report the minimum
 detectable odds ratio (or effect size) from the discordant-pair
 count, not from n; the two can move in opposite directions as data
 grows (§4.1: 6.0->8.0 at one cell as n went 467->750). If
 the claim is "no difference," run an equivalence test against a
 pre-specified bound and report the smallest bound achievable
 (§4.4: no cell reaches OR=2.0; achievable bounds 2.85-10.85).

-  **Does an alternative direction-estimation method change the
 conclusion?** A single estimator's failure to validate is ambiguous
 between "poor estimator" and "no effect"; checking at least one
 alternative distinguishes them (§4.3 checks logistic-regression
 weights and Fisher LDA).

Two further checks belong to passive probing rather than patching.

-  **Report a fold-seed sensitivity band alongside any
 cross-validated AUROC whose claimed effect is of comparable
 magnitude.** On this data the fold seed alone moves a component
 AUROC over a range of 0.055-0.100 (SD 0.013-0.022 over 50
 seeds) while the margins being compared are 0.003-0.011 (§4.5).
 Argmax peak-layer stability varies by architecture and must be
 checked, not assumed: over twelve seeds, GPT-2's FFN peak is
 perfectly stable (12/12 at L8) while its Attention peak splits
 two ways, Pythia's peaks take 3 and 5 distinct values, and
 Qwen0.5B's take 4 and 6. If a claim rests on which layer peaks,
 it needs a seed sweep before it can be made at all. A paired
 same-layer difference is much more stable and is the better estimand
 where the question allows it.

-  **If a grouped-CV diagnostic appears to collapse a probe's
 AUROC, run three checks before interpreting it.** A collapse is
 ambiguous between group leakage, a change of estimand, and failure
 to generalize across groups. (a) A permuted-group control, matched
 on both group size and per-group class composition, separates
 estimator behavior from real group structure. (b) Decompose the
 *standard* CV's own out-of-fold scores into within- and
 between-group pairs; grouped CV silently restricts the AUROC to
 within-group pairs, which on our data are only 4.9% of the pairs
 the pooled estimate is built from, so part of any "collapse" is a
 change of estimand rather than a change of performance. (c) A
 group-variable-only ceiling under the same CV protocol tests whether
 the group variable could carry the claimed effect at all -- and
 report both its in-sample and its cross-validated value, since the
 gap between them bounds what the check can conclude. In our case
 these moved the conclusion twice and left a claim narrower than
 either starting point (§4.10).

## 6. Discussion and Limitations

**What this paper does and does not claim.** We do not claim FFN
over-retrieval is present in closed-book generation, and we do not claim
it is absent. Each of our three instruments is compromised in a way we
can name: the passive probe's margins are smaller than its fold-seed
noise, the causal patch's direction cannot be validated at the available
held-out size, and its outcome metric is floored by model competence and
contaminated by degeneration. We claim only that this testbed and these
instruments cannot answer the question, and we give the checks that would
have to pass first.

**Scoped out: a RAG positive control.** The single most valuable
missing experiment is a positive control -- reproducing ReDeEP's original
asymmetry on a retrieval-augmented dataset with this exact pipeline, to
show that the pipeline *can* detect the effect where it is known to
exist. We did not attempt it, and our claims are scoped accordingly.
Without it, we cannot distinguish "the effect is absent in the
closed-book setting" from "this pipeline could not detect the effect
anywhere." We regard this as the correct next experiment for anyone
extending this line, and as a limitation that would need to be closed
before any positive or negative mechanistic claim.

**Scoped out: enlarging the judged pool further.** Enlarging GPT-2's
judged pool beyond the full 817-item validation split (e.g. by moving to
a different benchmark, or by sampling rather than greedy decoding) and
rerunning the causal kernel could further probe the competence-ceiling
finding. We did not do it, and the current thesis does not require it: the
point is that *this* testbed cannot answer the mechanistic question,
not that a larger sample of the same kind would rescue it -- and §4.1's
0/283 result is direct evidence that it would not.

**Also scoped out.** We did not run the grouped-CV or fold-seed
checks for the other five layer-localization methods in §4.9 (dense
probe, sparse probe, token-position probe, steering, logit lens). Given
§4.10's conclusion -- that the component probe does not generalize across
TruthfulQA topics -- the grouped-CV check on those five methods is now a
higher priority than we previously judged, not a lower one, and both
checks are cheap. We also did not test whether the cross-topic failure in
§4.10 is driven by within-topic near-duplicates or by non-transferring
topic-specific geometry; separating those would need a
paraphrase-controlled split we did not construct. We
make no comparison against black-box detectors (semantic entropy,
SelfCheckGPT), which are the production-relevant baseline. We do not
address benchmark contamination.

**Label validity.** All original results rest on a Jaccard
word-overlap label -- surface-form divergence rather than verified factual
incorrectness. We quantified this on all three architectures with an
independent LLM judge (Qwen2.5-3B-Instruct): a 100-item stratified GPT-2
sample gives 52% raw agreement, Cohen's kappa=0.04; full relabeling
of every completion gives kappa=0.032 (Pythia), 0.141
(Qwen0.5B-bare), and 0.084 (Qwen0.5B-chat) -- next to chance
throughout. Extending the relabel to all 534 GPT-2 completions gives
kappa=0.0417. The disagreement is consistently one-directional: the
judge calls far more completions hallucinated than the heuristic does
(on the full 534-item GPT-2 relabel the judge agrees with 97.0% of
Jaccard's hallucinated calls but only 7.1% of its correct calls,
computed directly from
`results/gpt2_full_534_judge_labels.json`), and manual reading
of disagreement cases confirms
the judge is usually right -- word overlap frequently credits completions
that share surface words with the reference but state a different specific
fact, or that are degenerate repetition loops, as "correct" (§4.2
quantifies the latter at 55.6% of Jaccard-correct completions).

A trivial length/lexical baseline check
(`code/32_surface_baseline_vs_judge_label.py`, the same
6-feature surface classifier rerun against the validated label on the same
cached features) gives logistic-regression CV AUROC =0.5604+/-0.1047
and MLP =0.5704+/-0.0971 (n=534, 5-fold). This is not chance-level,
and not meaningfully different from the same baseline's performance
against the Jaccard label (0.531/0.576) -- so the validated label is
not obviously *easier* for a surface-only classifier. That is weak
evidence against the alternative that the validated-label rise in
hidden-state probe AUROC (§4.6, roughly 0.60->0.70) is driven purely by
degeneracy detection, since a purely-degeneracy-driven account would
predict the surface baseline should also rise substantially. It is not a
strong exclusion: both baselines remain well short of the hidden-state
probes. §4.2's non-degenerate re-probe is the more direct test.

One residual limitation: the same judge model both defines the
found-direction's train split and scores every generated output in §4.4's
validated-label test, so that result's validity rests on the judge's own
accuracy, checked here only by manual spot-reading and the surface-feature
control -- not by an independent second judge or human annotation.

**Cheap baselines.** Before attributing discriminative power to the
FFN/Attention decomposition specifically, we checked whether trivial
generation-confidence signals do comparably well
(`code/41_cheap_baselines.py`, same n=534 validated label): an
undecomposed last-layer probe reaches AUROC =0.610+/-0.126;
teacher-forced generation-confidence features individually reach
0.54-0.64, with min-max-softmax strongest at 0.643+/-0.065, and all
four combined at 0.635+/-0.077. These sit in the same range as this
paper's FFN/Attention component probes (0.53-0.75 depending on layer
and label), so the specific decomposition studied here does not obviously
outperform a mechanism-agnostic confidence signal.

**No inference-economy claim.** This paper does not propose an
early-exit, routing, or compute-saving mechanism, and none of its AUROCs
are strong enough to gate anything at usable precision. We tested this
directly on the strongest passively-significant signal this project produced
(the SAE feature from §4.7's positive control, p=4.8x10^{-11}):
thresholding it as a single-feature classifier reaches AUROC =0.5614,
but the feature's raw activation is at or near zero for nearly every
sample, so no threshold above zero clears even 50% recall at any of the
four target-recall operating points tested and precision is pinned at the
4.8% base rate throughout -- extreme statistical significance under
simultaneous testing does not translate into usable gating concentration.

**Data and code availability.** All code, cached result JSONs, and
the paper source are included in the anonymized supplementary material for
double-blind review, and will be released as a public repository upon
publication. Every result that consumes only already-cached
`results/*.json` artifacts is self-contained. Four scripts
additionally require an unshipped sibling repository to *rerun from
scratch* (as opposed to re-verifying saved outputs):
`code/01_ffn_causal_patch.py` imports live code from a
`mech-int` sibling project to regenerate labeled completions;
`code/03_fisher_geometry_ffn_attn.py`,
`code/06_difficulty_matched_control.py`, and the feature-cache
step of `code/49_nondegenerate_subset_probe.py` depend on that
project's 2.9GB `activations.pkl`;
`code/15_sae_feature_gating_utility.py` reads cached hidden
states from a second sibling project. Cached intermediate feature files (`*.npz`) are excluded from the
released repository and regenerated by the scripts that consume them. One
consequence to state plainly: `code/50_cv_seed_sensitivity_sweep.py`
reads GPT-2 features from a cache written by
`code/49_nondegenerate_subset_probe.py`, which in turn requires
the unshipped `activations.pkl`; so `code/50`'s GPT-2 rows
inherit that same unshipped dependency for a from-scratch rerun, while its
Pythia and Qwen rows do not.

**Reproducibility map.** Entries are ordered by this paper's
emphasis: the validity checks first, the mechanism experiments second.

- **Competence ceiling (27/817; 0/283 newly judged)** (§4.1) `kaggle_kernels/paper1-causal-patch-enlarged-pool/` -> `kaggle_kernels/paper1-causal-patch-enlarged-pool/output/causal_patch_enlarged_pool_results.json`

- **Degeneracy check and non-degenerate re-probe** (§4.2) `code/04_degeneration_check.py`, `code/14_causal_patch_scaled_degeneration_filter.py`, `code/49_nondegenerate_subset_probe.py` -> `results/ffn_causal_patch_scaled_degeneration_filtered.json`, `results/nondegenerate_subset_probe.json`

- **Direction-validity gate: exact MDE/power table** (§4.3, §5) `code/51_direction_validity_mde_table.py` -> `results/direction_validity_mde_table.json`

- **Direction-validity gate, 200-resplit diagnostic** (§4.3) `code/46_direction_validity_resplit_diagnostic.py` -> `results/direction_validity_resplit_diagnostic.json`

- **Direction-validity gate, alternative estimators; one-/two-sided power** (§4.3) `code/44_alternative_direction_estimators.py`, `code/43_direction_validity_power_analysis.py` -> `results/alternative_direction_estimators.json`, `results/direction_validity_power_analysis.json`

- **Permuted-pseudo-category control and topic-only AUROC ceiling** (§4.10) `code/48_permuted_pseudocategory_control.py` -> `results/permuted_pseudocategory_control.json`

- **Original LOGO-CV diagnostic (interpretation revised by the controls above)** (§4.10) `code/47_category_leakage_diagnostic.py` -> `results/category_leakage_diagnostic.json`

- **CV fold-seed sensitivity sweep and gap decomposition** (§4.5) `code/50_cv_seed_sensitivity_sweep.py` -> `results/cv_seed_sensitivity_sweep.json`

- **Causal patching, base runs** (§4.4) `code/01_ffn_causal_patch.py`, `code/10_ffn_causal_patch_scaled.py` -> `results/ffn_causal_patch_results.json`, `results/ffn_causal_patch_scaled_results.json`

- **Causal patching under the validated label** (§4.4) `kaggle_kernels/paper1-causal-patch-judge-label/` -> `results/causal_patch_judge_label_results.json`
- **Causal patching beyond GPT-2 (Pythia, Qwen0.5B-chat)** (§4.4) `code/07_multi_arch_causal_patch.py` -> `results/multi_arch_causal_patch.json`

- **Component-specificity test with a real Attention direction** (§4.4) `kaggle_kernels/paper1-causal-patch-real-attn-direction/` -> `results/causal_patch_real_attn_direction_results.json`

- **Permutation cosine null, common-site test, random-direction ensemble** (§4.4) `kaggle_kernels/paper1-causal-patch-tier1-validated/` -> `kaggle_kernels/paper1-causal-patch-tier1-validated/output/causal_patch_tier1_validated_results.json`

- **TOST equivalence bounds, competing-risks table** (§4.4) `code/38_tier1_tost_competing_risks.py` -> `results/tier1_tost_competing_risks.json`

- **Low-dose alpha sweep** (§4.4) `code/42_low_dose_alpha_sweep.py` -> `results/low_dose_alpha_sweep.json`

- **FFN/Attn dosage diagnostic** (§4.4) `code/26_ffn_attn_dosage_diagnostic.py` -> `results/ffn_attn_dosage_diagnostic.json`

- **FFN vs. Attention component decomposition** (§4.5, §4.6) `code/02_cross_arch_component_probe.py` -> `results/cross_arch_component_probe_*.json`

- **Paired DeltaAUROC and nested-CV selection-bias check** (§4.5) `code/37_paired_component_delta_auroc.py` -> `results/paired_component_delta_auroc.json`

- **Qwen chat-template reversal** (§4.6) `code/02_cross_arch_component_probe.py qwen05chat` -> `results/cross_arch_component_probe_qwen05chat.json`

- **ROME-style causal tracing** (§4.7) `code/08_rome_style_causal_tracing.py`, `code/09_multi_arch_rome_style_causal_tracing.py`, `code/18_rome_style_causal_tracing_scaled.py`, `code/40_rome_style_causal_tracing_validated.py` -> `results/rome_style_causal_tracing*.json`

- **SAE feature clamp (GPT-2 null; companion positive control)** (§4.7) generating script for the GPT-2 null not released (predates current numbering); `code/15_sae_feature_gating_utility.py` -> `results/sae_feature_clamp_paper1.json`, `results/sae_feature_gating_utility.json`

- **Difficulty-matched and adversarial controls** (§4.8) `code/06_difficulty_matched_control.py`, `code/11_multi_arch_difficulty_matched_control.py`, `code/17_gradient_reversal_adversarial_probe.py`, `code/30_difficulty_matched_control_judge_label.py` -> `results/difficulty_matched_control*.json`, `results/multi_arch_difficulty_matched_control.json`, `results/gradient_reversal_adversarial_probe.json`

- **Layer localization (7 methods), Jaccard and validated label** (§4.9) `code/00_verify_vendored_mechint_numbers.py`, `code/29_gpt2_full_validated_relabel_rerun.py` -> `results/vendored_mech_int/`, `results/gpt2_full_validated_relabel_rerun.json`

- **Surface-feature baseline under the Jaccard label (the 6 features, and the 0.531/0.576 figures)** (§4.8, §6) `code/05_run_surface_baseline.py`, `code/05_surface_baseline_classifier.py` -> `results/surface_baseline/`
- **Label-validity audit and surface baselines** (§6) `code/16_llm_judge_label_noise.py`, `code/23_regenerate_completions_for_judge.py`, `code/24_llm_judge_score_all_architectures.py` (run as `kaggle_kernels/paper1-llm-judge-relabel/`), `code/32_surface_baseline_vs_judge_label.py`, `code/41_cheap_baselines.py` -> `results/llm_judge_label_noise.json`, `results/llm_judge_relabel_summary.json`, `results/surface_baseline_vs_judge_label.json`, `results/cheap_baselines.json`

- **GPT-2 full-534 judge relabel** (§6) `kaggle_kernels/paper1-gpt2-full-judge-relabel/`, `code/28_judge_label_all_gpt2_534.py` -> `results/gpt2_full_534_judge_labels.json`
- **Figure 1** (§4.6) `code/20_generate_ffn_attn_figure.py` -> `draft/latex/figures/ffn-attn-comparison.pdf` (plots already-reported numbers; no new computation)

## 7. Conclusion

We set out to test whether an FFN-vs-Attention hallucination asymmetry
analogous to ReDeEP's appears in closed-book generation on small models.
The answer this paper can support is that the question is not answerable
on this testbed with these instruments, and we can now say precisely why.

GPT-2 answers 3.3% of TruthfulQA validation items correctly under an
independent judge, and adding 283 previously-unjudged items added zero
correct answers -- so the flip-to-correct outcome variable is floored
regardless of any mechanism. Over half of the nominally hallucinated
completions in each causal-test pool we constructed (51.9% at n=81,
53.1% at n=228; 53.6% of all 534 baseline completions
regardless of class) are degenerate repetition loops rather than
confabulations, near-balanced across label classes, so the outcome events
that do occur are confounded with loop-breaking. And the difference-of-means direction the causal test
injects never cleared a held-out validity check -- one which, at the
n=11 holdout this testbed supports, cannot declare any *observed*
AUROC below 0.875 significant, has power 0.31 against a true AUROC of
0.75, and passes a pure-noise direction 6.7%-46.1% of the time
depending on where its threshold is set.

We therefore do *not* report the causal null as surviving scrutiny.
The causal instrument does not provide interpretable evidence either way,
because the directions it patches never passed the validity gate. That is
a limitation of the instrument, stated as such.

The passive side is weaker than an earlier version of this work claimed,
in a different way. The peak-versus-peak FFN-vs-Attention margins
(0.003-0.011 AUROC) are three to ten times smaller than the spread a
single cross-validation fold seed produces on the same features, and on
both 24-layer models the argmax "peak layer" is not a well-identified
quantity at all -- it lands on three to six different layers across twelve
seeds. A same-layer paired comparison is stable, but its sign flips
between layers within the same architecture, so which layer is compared
determines the answer. And a leave-one-category-out re-test, with two
permutation controls, a pair-type decomposition, and a
group-variable-only ceiling calculation, establishes that a standard
random K-fold AUROC for this probe on TruthfulQA overstates
unseen-topic performance by 0.13-0.17 AUROC, with cross-topic
performance at the two layers tested at chance -- while leaving the mechanism
behind that gap unidentified (§4.10).

What we offer instead is transferable: two validity checks (a competence
ceiling measurement and a degeneracy pre-filter) that are cheap,
model-agnostic, and would have changed how we read our own flagship
result had we run them first; an exact minimum-detectable-effect
calculation for held-out direction validation that replaces "n is
small" with a number; and a checklist assembling these with the controls
a component-specificity patching claim requires. Any researcher running an
intervention on an LLM and reporting a flip-rate inherits these same
risks, independent of which component or behavior is being tested.

## 8. Appendix A: Correction History

This appendix is a transparent audit trail of errors found during
iterative internal and external review, and of the checks run to confirm
each fix's effect on reported numbers. The main text states final,
corrected numbers directly; nothing below is required to verify them.

-  **A fabricated-looking claim, removed.** An earlier version
claimed "a direct FFN-found-vs-Attention-found comparison gives McNemar
p=1.000 in every configuration." This was never computed:
`code/01_ffn_causal_patch.py` and its scaled variants save an
Attention-found flip rate but never run a McNemar test against it, and in
any case that condition patches the FFN-derived direction at the
Attention site rather than a direction derived from Attention's own
activations, so no version of that script could have supported an
FFN-vs-Attention specificity claim. The genuine test (§4.4) uses a real
Attention-derived direction. A related paragraph's degeneracy rate
("19.8-29.6% across conditions") was similarly only the FFN-found
arm at L8 from an earlier, smaller pass, not the full range across all
twelve conditions it was presented as describing; the correct range
(10.1%-46.5%) is what §4.4 now reports.

-  **Judge-parser substring bug, fixed and checked.** Every
judge-scoring function in this project originally classified a verdict as
"correct" if the string `"CORRECT"` appeared and
`"HALLUCINAT"` did not -- and `"CORRECT"` is a substring of
`"INCORRECT"`, so a verdict of "INCORRECT" would have been
silently scored as label 1 in seven separate implementations. We had
never persisted raw verdict strings, so this was unfalsifiable from
existing artifacts until a review flagged it. Fixed in all seven
implementations by checking for `"HALLUCINAT"` and
`"INCORRECT"` first, then reran the full 534-sample GPT-2
relabel end to end: byte-identical to the original (same 27/507
split, same kappa=0.0417). We additionally reran the §4.4 flagship
causal test (467 test prompts x 17 scored conditions -- §4.4's
sixteen patched conditions plus the unpatched baseline) with the
corrected parser and compared against the then-current saved per-sample
labels: identical at every one of the 467x17 scored cells. One
caveat a reader should know: the pre-fix run's artifacts were overwritten
rather than retained, and raw judge verdict strings were never persisted
at all, so this particular comparison cannot be re-verified from the
released supplement -- only the post-fix artifacts are shipped. The bug
was real and needed fixing, and on our own check changed no reported
number.

-  **Dosage-mismatch diagnostic, corrected twice.** An earlier
version of `code/26_ffn_attn_dosage_diagnostic.py` (a) measured
on bare TruthfulQA questions rather than the
`"Q: {question} nA:"` prompts the causal test
patches, and (b) normalized alpha against each sublayer's own raw
output norm -- the wrong denominator, since the patching hook replaces
that output with (out+alpha) and the block's
own forward code then adds the combined value into the residual stream.
The uncorrected version reported a 2-3x dosage asymmetry between
arms; correcting only the prompt format shrinks it to 1.3-1.5x;
correcting both gives the 16.0%/12.0% (alpha=20) and
32.1%/24.1% (alpha=40) numbers in §4.4. A further correction, made
in this round: because the corrected script uses one shared
residual-stream denominator for both arms, the resulting arm-to-arm
equality is true by construction and is now labeled as such in §4.4
rather than presented as an empirical finding of "no dosage asymmetry."

-  **Surface-baseline computation, flagged as missing.** An
earlier version of the label-validity discussion asserted that a trivial
length/lexical baseline does not explain the judge label's structure at
chance-level AUROC -- no such computation existed in this project at the
time. `code/32_surface_baseline_vs_judge_label.py` was written
to actually compute it; the result (not chance-level, and not meaningfully
different from the same baseline against the Jaccard label) is reported in
§6.

-  **Direction-validity CIs misdescribed, and a 10x cosine
transcription error.** (a) The four direction-validity bootstrap CIs at
n=11 were described as "all consistent with chance"; three of the four
in fact exclude 0.5 entirely, and an exact Mann-Whitney test finds them
nominally significant in the anti-predictive direction (§4.3); a further
correction this round made those p-values consistently two-sided (L8
FFN/Attn 0.0485, not the one-sided 0.0242 quoted before). (b) The
permutation-based cosine-similarity check reported "-0.058/-0.051";
the underlying result file gives -0.0058/-0.0051. To avoid a
confusion this correction has caused before: that -0.0058/-0.0051 is
the tier-1 kernel's own pair of directions (fit on 47 items), and is a
different quantity from the -0.054/-0.056 reported in §4.4 and §5,
which is the cosine between the FFN and the genuinely Attention-derived
direction in the component-specificity kernel (fit on 58 items). Both
are correct for their own run; neither supersedes the other. Fixing (a) exposed
that the power analysis used to argue the test was uninformative
(`code/43`) was itself one-sided, structurally unable to detect
power in the direction the data fall in. Recomputed two-sided, the test is
reasonably well-powered (73-99%) at the observed AUROCs, which
motivated the 200-resplit diagnostic
(`code/46_direction_validity_resplit_diagnostic.py`) reported in
§4.3. That diagnostic resolved the question: the original seed's
anti-predictive result was an atypical low-tail draw, not a stable
property of these directions.

-  **Random-direction-ensemble percentiles, misread as
informative.** An earlier version described the random-direction ensemble
as "the strongest single check" and reported the found directions at the
"50th" and "40th" percentile of a 20-draw ensemble. Both the framing
and the reading were wrong: the found directions produced 0/60 flips,
and 10 of 20 random FFN directions and 8 of 20 random Attention
directions also produced 0/60, so those percentiles are tie counts at
the statistic's floor. The same applies to the enlarged-pool rerun's
"95th"/"100th" percentiles, which are driven by 2/60 and 3/60 flip
events. §4.4 now reports raw counts and states that this check is
uninformative at this flip-rate floor.

-  **A CV-seed effect misattributed to an aggregation
convention.** An earlier version explained the gap between two AUROCs
reported for the same GPT-2 L8 FFN features (0.6053 and 0.643) as a
difference between mean-of-folds and pooled out-of-fold aggregation, and
separately explained a peak-layer disagreement for Qwen0.5B as the two
aggregation conventions "genuinely disagreeing." Both attributions were
wrong. `code/02` uses `StratifiedKFold(random_state=42)`
and `code/37` uses `random_state=0`; the seed accounts for
+0.0365 of the +0.0374 total gap and the aggregation convention for
+0.0010 (`code/50_cv_seed_sensitivity_sweep.py`). §4.5 now
reports the decomposition and the resulting fold-seed sensitivity band,
and attributes the peak-layer instability to seed sensitivity.

-  **A category-leakage claim, twice revised and finally
narrowed.** An earlier version reported, in its abstract and headline
results, that a leave-one-category-out CV re-test collapsed the component
probe's AUROC from 0.62-0.66 to 0.48-0.49, attributed this to
category-clustering leakage (specifically, the probe learning per-topic
correct-answer rates), and extended the claim to a "heterogeneous"
across-architecture leakage result. An independent review then argued the
entire effect was an artifact of the LOGO estimator's per-category
averaging at n_+~1.7. We ran the controls that separate these
(`code/48_permuted_pseudocategory_control.py`); *both*
readings turned out to be wrong, in different directions.
(a) Two permutation controls (size-matched, and size- and class-matched)
show random groupings recover 0.58-0.62, not chance, so the estimator
is not biased downward at this n and the collapse is specific to real
topic structure (0-2 of 100 matched draws fall at or below the real
value across the four cells). We predicted the artifact outcome before
running this control and were wrong.
(b) A group-variable-only ceiling calculation shows the originally claimed
mechanism cannot be demonstrated: topic identity alone yields AUROC
0.7938 in-sample but 0.5054 leave-one-out and 0.4863/0.4789 under
this paper's own 5-fold CV. We initially wrote this up as ruling the
mechanism out; that too was an overreach, since the in-sample value
exceeds the probe's own AUROC and the shortfall out of fold is an
estimation-noise problem at 27 positives over 38 categories, not a
demonstration that topic base rates carry no signal. The text now says
only that the account cannot be tested here.
(c) A third check, added last, decomposes the standard CV's own
out-of-fold scores by pair type and shows that LOGO silently changes the
estimand: only 4.9% of standard-CV pairs are within-topic, and
within-topic AUROC on the same scores is 0.536-0.684 rather than
chance. Part of the "collapse" is therefore this change of estimand and
part is the removal of same-topic training data; §4.10 separates them.
The surviving claim is that standard random K-fold CV overstates
unseen-topic performance for this probe by 0.13-0.17 AUROC, with the
mechanism unidentified. The cross-architecture extension is removed
regardless, being confounded by three simultaneous protocol differences
aliased with architecture.

-  **Small numeric and labeling corrections made in this round.**
(a) The GPT-2 534-item judge-vs-Jaccard kappa was stated as 0.03 in
one place; the correct value is 0.0417 (0.03 is Pythia's kappa,
mislabeled) (§3, §6). (b) Pythia's Jaccard-label peak FFN AUROC was stated as
0.615 in one place and 0.6181 in another; the artifact-verified value
is 0.6181 (§4.6). (c) Two different minimum-detectable-odds-ratio ranges
(3.75-8.0 and 3.25-6.0, both at n=467) were presented as though
they described the same test; they are the native-site
FFN-vs-Attention comparison (directions fit on 58 items) and the
common-site comparison (directions fit on 47 items) respectively, and
are now labeled (§4.1, §4.4). (d) The TOST equivalence-bound range (§4.4) was
stated as 2.85-10.9; the maximum achieved in the underlying grid
search is 10.85 (`results/tier1_tost_competing_risks.json`).
An earlier version of this correction added that "10.9 is not a grid
point"; that justification was itself wrong -- `code/38` searches
`np.arange(1.05, 15.0, 0.05)`, which does contain 10.90. The
number is simply not the one the search returns. (e) The low-dose sweep range (§4.4) was
stated as 42-52%; the computed range is 41.67-51.67%, and the
claim of "no monotone dose-response" was inaccurate -- the Attention arm
is strictly monotone decreasing across the three doses, so the accurate
statement is "no monotone increase." (f) The enlarged-pool replication
was described as "nearly double the power" at n=750 versus n=467;
the ratio is 1.61x, and (per §4.1) discordant-pair counts, not n,
determine resolution.

-  **Correction confessions relocated.** Earlier drafts narrated
several of the above corrections inline in the results sections. All are
now collected here, and the main text states final numbers directly; where
a reader might otherwise reconstruct the wrong quantity, §4.3, §4.4, §4.5
and §5 carry a one-clause pointer to the relevant item above rather than a
retelling. The single deliberate exception is §4.10, whose subject
*is* how its own interpretation changed under two permutation
controls, a pair-type decomposition, and a ceiling calculation; that
history is reported in the body because the sequence of checks, not the
final number alone, is the transferable result.

## 9. References

Full citations below, compiled from exactly the bibliographic detail
already verified in-text or in `related_work/related_work_notes.md`;
entries with no author list recorded anywhere in this project are cited
by title only rather than inventing names.

Sun, Y., et al. (2025). ReDeEP: Detecting Hallucination in
Retrieval-Augmented Generation via Mechanistic Interpretability. *ICLR
2025*. arXiv 2410.11414.

ParamMute. *NeurIPS 2025*. arXiv 2502.15543.

SEReDeEP. arXiv 2505.07528.

FACTUM. arXiv 2601.05866.

Xiong, et al. RAGLens: Toward Faithful Retrieval-Augmented Generation
with Sparse Autoencoders. arXiv 2512.08892.

Detection Without Correction: A Robust Asymmetry in Activation-Based
Hallucination Probing. arXiv 2604.13068.

Meng, K., Bau, D., Andonian, A., & Belinkov, Y. (2022). Locating and
Editing Factual Associations in GPT. *NeurIPS 2022*. arXiv 2202.05262.

Meng, K., Sharma, A. S., Andonian, A., Belinkov, Y., & Bau, D. (2023).
Mass-Editing Memory in a Transformer. *ICLR 2023*. arXiv 2210.07229.

Dai, D., Dong, L., Hao, Y., Sui, Z., Chang, B., & Wei, F. (2022).
Knowledge Neurons in Pretrained Transformers. *ACL 2022*. arXiv
2104.08696.

A Single Direction of Truth. arXiv 2507.23221.

Kantamneni, et al. Are Sparse Autoencoders Useful? A Case Study in
Sparse Probing. arXiv 2502.16681.

Li, K., Patel, O., Viégas, F., Pfister, H., & Wattenberg, M. (2023).
Inference-Time Intervention: Eliciting Truthful Answers from a Language
Model. *NeurIPS 2023*. arXiv 2306.03341.

Kuhn, L., Gal, Y., & Farquhar, S. (2023). Semantic Uncertainty:
Linguistic Invariances for Uncertainty Estimation in Natural Language
Generation. *ICLR 2023*. arXiv 2302.09664.

Manakul, P., Liusie, A., & Gales, M. (2023). SelfCheckGPT: Zero-Resource
Black-Box Hallucination Detection for Generative Large Language Models.
*EMNLP 2023*. arXiv 2303.08896.
