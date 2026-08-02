# Uninterpretable Nulls in Interpretability Intervention Studies: A Closed-Book Case Study and a Validity Checklist

**Anonymous Author(s)**
Paper under double-blind review

## Abstract

Interpretability intervention studies routinely report a null -- the patch
changed nothing, or nothing differentially between two components -- as
evidence about a mechanism. We show that such a null can be uninterpretable
for reasons that are cheap to check in advance and are almost never checked,
and we supply the checks: a testbed *competence ceiling*, a *degeneracy
contamination* of the outcome metric, and an exact power calculation for
held-out *direction validity*, assembled into a ten-item validity checklist (8
items for causal-patching studies, 2 for passive probing) with a table stating
when it binds and when it does not. Our case study is a closed-book test of
whether small models (GPT-2, Pythia-410M, Qwen2.5-0.5B) show an
FFN-vs-Attention hallucination-detection asymmetry analogous to the
retrieval-augmented one ReDeEP reports. All three checks fire, and together
they show the intended target hypothesis cannot be tested on this testbed.

**Competence ceiling.** GPT-2 answers 27/817 (3.3%) of TruthfulQA validation
items correctly under an independent LLM judge, and 0 of the 283 items added
to enlarge that pool were judged correct -- though those 283 had been
pre-selected for near-zero word overlap with every reference answer (§4.1),
making 0/283 weaker evidence than an unconditioned one. Any flip-to-correct
metric is therefore floored near zero regardless of the underlying mechanism.

**Degeneracy contamination.** Over half the nominally hallucinated completions
in each of our pools (51.9%, 53.1%, and 53.6% of all baseline completions in
the full labeled pool) are degenerate repetition loops rather than
confabulations, caught by one cheap model-agnostic check, and degeneration is
near-balanced across the word-overlap (Jaccard) label's classes (55.6% versus
51.5%), so it contaminates both arms of a flip-rate comparison. Removing every
degenerate item does not collapse the passive probe under that label -- its
AUROC survives and slightly rises -- so degeneracy-propensity does not suffice
to explain the passive signal; under the judge label the same subset leaves
only 17 positives and is uninformative either way. We report both.

**Direction validity, calculated exactly.** A difference-of-means direction
fit at this sample size fails a held-out validity check we formalize and give
an exact power calculation for, as a function of both class counts. At the 3:8
holdout the causal kernel used, the null arises from only C(11, 3)=165
equiprobable label arrangements; no *observed* AUROC below 0.875 can ever be
significant at one-sided alpha=0.05; and a pure-noise direction passes
plausible gates 6.7%-13.9% of the time. That 8 was a per-class cap, not a
property of the data -- 475 of the pool's 507 judge-hallucinated items were
never used to fit the direction -- and re-running the gate against all of them
improves every figure at zero data cost (MDE 0.875to0.779) and removes an
apparent anti-predictive result that was an artifact of the small negative
holdout. *Positives* bind: 27 judge-correct items exist in total.

**Fold-seed sensitivity, on the passive side.** Changing only the
cross-validation fold seed moves a component's AUROC over a range of
0.055-0.100 (SD 0.013-0.022 over 50 seeds), an order of magnitude larger than
the peak-versus-peak FFN-vs-Attention margins at issue (0.003-0.011), and the
argmax "peak layer" lands on three to six different layers across twelve seeds
on both 24-layer models. The paired same-layer difference is stable, but its
sign flips between layers within an architecture.

**A grouped-CV collapse, resolved.** We report a leave-one-category-out (LOGO)
diagnostic whose interpretation changed twice under scrutiny, since the
resolution is the methodological point. Permutation controls matched on group
size and per-group class composition show the collapse (0.62-0.66 under
standard CV to 0.48-0.49) is specific to real topic structure and *not* an
artifact of per-category averaging, at every cell tested and under
Holm-Bonferroni correction; the collapse comes from LOGO restricting
comparisons to within-topic pairs, only 4.9% of standard-CV pairs. Holding the
averaging convention fixed, the residual from removing same-topic items from
training is indistinguishable from zero, and this probe's within-topic
discrimination is at chance either way (0.44-0.55). An earlier version
reported a 0.06-0.20 "training overlap" effect that was entirely an artifact
of subtracting a category-averaged number from a pair-weighted pooled one, and
a ceiling calculation shows the originally claimed mechanism -- the probe
reading off per-topic correct-answer rates -- cannot be demonstrated. What
survives is narrower than either earlier reading: *a standard random K-fold
AUROC for this probe on TruthfulQA overstates its unseen-topic performance by
roughly 0.13-0.17 AUROC, and cross-topic performance at the layers tested is
chance*, with the mechanism unidentified.

**Scope.** We do not claim FFN over-retrieval is present or absent in
closed-book generation. We show that this testbed and instrument cannot
currently answer that question, and provide the checks that would need to pass
before a claim either way could be trusted.

## 1. Introduction

A large fraction of mechanistic-interpretability claims rest on an
intervention: patch an estimated direction or activation into a model, measure
whether the model's output changes in a specified way, and read the result as
evidence about a mechanism. When such an experiment returns a null -- the
intervention changed nothing, or changed nothing differentially between two
components -- that null is usually reported as evidence about the mechanism.
This paper is about two ways that reading can be wrong, both of which we hit
directly, and neither of which is routinely checked.

The first is a **competence ceiling**. A flip-to-correct metric can only move
if the model is capable of producing the correct answer at all; if the base
model is near-zero-accuracy on the testbed, the metric is floored by
construction and every downstream comparison inherits that floor. The second
is an **outcome-metric contamination**. "The generation changed from wrong to
right" measures semantic correction only if the baseline generations are wrong
*answers*; if a large share are degenerate repetition loops, an intervention
that "fixes" them may only have broken the loop -- a generic perturbation
effect, not a targeted semantic one. Both failure modes are cheap to detect,
expensive to ignore, and invisible in the summary statistics such papers
usually report. We found them while trying to answer a specific mechanistic
question, and that question is now the least reliable thing in this paper; the
diagnostics are the most reliable.

**The question we set out to answer.** ReDeEP (Sun et al., ICLR 2025)
localized one mechanism by which fluency and factuality diverge in
retrieval-augmented generation: FFN sublayers overriding retrieved context, a
"Knowledge FFN" over-retrieval pattern. We asked whether an *analogous*
asymmetry -- FFN sublayer activity relating more strongly to hallucination
than Attention sublayer activity -- is detectable in pure closed-book
generation, where there is no retrieved context to override. This is not a
direct test of ReDeEP's mechanism, which is defined over response tokens with
retrieved context present; every passive probe here reads the model's forward
pass over the *prompt only*, before generation (§3). These are different
questions, and we describe our results only in terms of the question we
actually ran.

**What we found instead.** The intended test is not answerable on this
testbed, for three separable reasons -- each a check any comparable study
could run cheaply: the outcome variable barely exists, at 27/817 (3.3%)
correct under an independent LLM judge with the 283 previously-unused items
adding exactly zero (§4.1); over half the completions in each of our three
pools are repetition loops rather than confabulations, near-balanced across
label classes, so they contaminate both arms of a flip-rate comparison (§4.2);
and the difference-of-means direction the causal test patches in cannot be
validated at the 3:8 held-out split the causal kernel used, whose per-class
cap rather than any data limit set that size, with the 27 judge-correct items
binding (§4.3).

Given all three, the causal null we do observe (no FFN-vs-Attention difference
surviving correction at any tested configuration, under two labels, at two
injection sites, at n=467 and n=750 -- the single nominal exception, a
common-site p=0.049, does not survive Holm correction across even its own four
tests) is not evidence that the two components are equally causal. It is what
an uninformative instrument returns.

**Contributions.**

-  **A competence-ceiling diagnostic** (§4.1), obtained by judging every item
   in the benchmark split rather than only the subset an earlier heuristic
   could score. It shows the testbed, not the sample size, is the binding
   constraint, converting "we need more data" into "more data of this kind
   cannot help."

-  **A degeneracy pre-filter for flip-rate metrics** (§4.2). One
   model-agnostic check -- a 4-, 5-, 6- or 8-word phrase repeated 3+ times --
   flags over half of this pool's "hallucinations." We report its rate on
   three pools, its balance across label classes, and its effect on both the
   causal test and the passive probe.

-  **A direction-validity gate with an exact minimum-detectable-effect table**
   (§4.3, §5), replacing "n is small" with the exact null distribution of a
   held-out AUROC over all binomn_++n_-n_+ label arrangements, tabulated as a
   function of both class counts.

-  **A ten-item validity checklist** (§5) -- eight for causal-patching
   studies, two for passive probing -- distilled from the above plus the
   controls this paper's own causal section required, with a table stating
   when it binds and when it does not. It is the exportable artifact here,
   written to be read independently of anything about FFNs.

-  **A fold-seed sensitivity result for passive probes** (§4.5), including an
   exact decomposition of a 0.037 AUROC discrepancy this repository itself
   contained, and evidence that the argmax "peak layer" is not a
   well-identified quantity at this sample size.

-  **A worked case study in interpreting a grouped-CV "collapse"** (§4.8), in
   which two permutation controls, a pair-type decomposition and a
   group-variable-only ceiling calculation tell against both the leakage
   reading and the estimator-artifact reading, and leave a narrower claim:
   standard K-fold CV overstates unseen-topic performance by 0.13-0.17 AUROC,
   by a mechanism these experiments cannot identify.

We deliberately do not claim a mechanistic finding. The peak-versus-peak
margins this literature compares are 0.003-0.011 AUROC on our data -- three to
ten times smaller than the spread a single cross-validation fold seed produces
on the same features (+/-0.03 range, 0.037 between the two seeds this
repository happened to use, §4.5) -- and the causal instrument does not clear
its own validity gate. What we can support is the negative methodological
claim, and the checks that would need to pass before a positive one could be
made.

## 2. Related Work

**Parametric factual knowledge in FFN sublayers.** ROME (Meng et al. 2022) and
MEMIT (Meng et al. 2023) causally locate and edit specific facts by writing to
mid-layer FFN weight matrices; Knowledge Neurons (Dai et al. 2022)
independently identifies FFN-resident neurons whose activation correlates with
specific factual recall. Our claims are far weaker and of a different type: we
do not establish an FFN-resident mechanism, and we do not claim to have
refuted one. We report that a particular closed-book instrument cannot decide
the question.

**RAG-scoped FFN over-retrieval.** ReDeEP (Sun et al., ICLR 2025) reported FFN
override of retrieved context during RAG hallucination. Follow-up work (Huang
et al. 2025; Wang 2025; Dassen et al. 2026; Xiong et al. 2025) refines this
mechanism or extends it with sparse-autoencoder tooling, but all remain
RAG-scoped. Our setting is closed-book and prompt-only (§3), so it is at best
an analogue of theirs, not a replication attempt, and a null here says nothing
about the RAG result.

**Detection without correction.** An independent finding (Roy et al. 2026)
reports that output-confidence baselines beat activation probes above 410M
parameters, and that residual-stream steering flips 0/7 tested models'
generated answers toward correct across 42 configurations on GPT-2-scale
models. Our causal section reaches a compatible null at the component level,
but we are explicit that our version of that null is uninterpretable for the
reasons in §4.1-§4.3 rather than being an independent confirmation.

**Black-box detection.** Semantic entropy (Kuhn et al. 2023) and SelfCheckGPT
(Manakul et al. 2023) operate purely on output text and generalize to closed
API-only models where hidden states are unavailable -- the actual
production-relevant baseline mechanistic probes would need to beat. We make no
such comparison here and flag its absence explicitly.

**Sparsity and steering method provenance.** The 100/768-dimension sparse
probe result (Appendix F) is consistent with Kantamneni et al. ("Are Sparse
Autoencoders Useful? A Case Study in Sparse Probing," arXiv 2502.16681), which
finds dense L1 probes often match or beat trained SAE probes; it is not
evidence for a claim made by the superficially similar O'Neill et al. (2025),
which does not make this comparison. The difference-of-means steering method
used throughout descends from Inference-Time Intervention (Li et al. 2023,
arXiv 2306.03341).

**Validity checklists and audits.** The closest relatives of this paper's
contribution are not mechanism papers but protocol audits. Hewitt and Liang
(2019) set the template for probing: a probe's accuracy is uninterpretable on
its own, and what licenses a claim is its *selectivity* against a control task
the same probe is fit on. Two recent audits make the analogous point for
hallucination detection. Janiak et al. (2025) show that reported detection
AUROCs depend heavily on the correctness label used -- swapping a
lexical-overlap metric (ROUGE) for an LLM judge drops several established
methods by up to 45.9% -- the same substitution §6 makes, on the same kind of
label (word-overlap Jaccard versus an LLM judge, kappa=0.04); on our data the
swap moves the probe's AUROC *up* rather than down (§4.6), which we read as
the two labels being noisy about different things rather than as a
disagreement about the audit's point. Hussain and Kantarcioglu (2026) show
that four of six widely-used hallucination-detection corpora embed the
reference answer in the model's input, so a text-similarity baseline with no
access to model internals scores near-perfectly, and that only three of the
many previously reported high-AUROC cells survive their verification checks.
Our checklist is scoped complementarily to both: it targets intervention
experiments (patching an estimated direction and reading a flip-rate) rather
than the passive detection benchmarks those audits cover.

## 3. Setup and Scope

**Models, data, labels.** We use GPT-2 (124M), Pythia-410M, and
Qwen2.5-0.5B-Instruct on the TruthfulQA generation validation split.
Completions are greedy-decoded (40 new tokens) from a bare "Q: {question}\
nA:" prompt (and, for Qwen0.5B, a chat-templated variant, §4.6). Two labels
are used throughout and always named explicitly: a *Jaccard* word-overlap
heuristic against TruthfulQA's reference answers, and a *validated* label from
an independent LLM judge (Qwen2.5-3B-Instruct). The two labels agree only
52.2% of the time on GPT-2 (kappa=0.0417, on the full 534-item relabel);
per-architecture kappa is 0.032 (Pythia), 0.141 (Qwen0.5B-bare), and 0.084
(Qwen0.5B-chat). §6 discusses what this does and does not license.

**A disclosure that scopes every passive result in this paper.** Every passive
probe and every estimated direction reported here is computed from the model's
forward pass over the *prompt alone*, never over any token of the model's own
generated completion (`mech-int/src/extraction/activations.py` runs one
forward pass on the prompt string; `code/02_cross_arch_component_probe.py`
hooks a single `model(**inputs)` on the same). The correct-vs-hallucinated
label is a property of a separately-generated completion whose tokens never
pass through the model for these measurements. So every passive number here
answers: *does the model's pre-generation internal state predict whether its
own upcoming answer will be judged correct?* That is not the same question as
probing an already-generated hallucinated answer for features of its content,
and it is not ReDeEP's question. Readers should not infer from "last-token
activation" or "hidden-state probe" language that completion tokens were
involved.

**Two signatures, defined in advance.** For the closed-book analogue we set
out to test, two independent signatures would jointly constitute a positive
result: (a) a *passive* signature -- FFN-derived features detect the label
with reliably higher AUROC than Attention-derived features at the two
components' respective peak layers (§4.5-§4.6); and (b) an *active* signature
-- patching an FFN-derived direction into the FFN sublayer during generation
corrects more hallucinations than an equivalent, genuinely Attention-derived
direction patched into the Attention sublayer (§4.4). Neither is observed
reliably. §4.1-§4.3 explain why, for (b), "not observed" does not mean
"absent."

## 4. Results

The three checks that carry this paper come first: the competence ceiling
(§4.1), the degeneracy contamination (§4.2), and the direction-validity gate
(§4.3). §4.4 then reports the causal null those three make uninterpretable.
§4.5 is the passive-side counterpart -- fold-seed sensitivity, which is the
single strongest evidence here that the peak-versus-peak FFN-vs-Attention
comparison is not measuring anything -- and §4.6-§4.8 report the remaining
passive and causal instruments, ending with a worked resolution of a
grouped-CV collapse (§4.8).

### 4.1 The testbed's competence ceiling

Every causal-patching result in this paper measures a flip-to-correct rate:
the fraction of prompts whose baseline generation is labeled hallucinated and
whose patched generation is labeled correct. That metric is bounded above by
the model's ability to produce a correct answer at all.

Under the validated judge label, GPT-2 is correct on 27 of 534 items (5.1%) in
the pool the original analyses used -- itself the subset of TruthfulQA's
817-item validation split that an older Jaccard-label filter could score, 283
items having been discarded before the judge existed. We judged all 283
(`kaggle_kernels/paper1-causal-patch-enlarged-pool/`): **zero were judged
correct**, leaving the full-split rate at 27/817 = 3.3%, exactly unchanged
from the 27 correct in the original subset.

Those 283 were not a random sample, and the selection rule matters: an item
was dropped exactly when its completion's maximum Jaccard word-overlap was at
or below 0.12 against *every* correct reference answer *and* at or below 0.12
against every incorrect one (`code/23_regenerate_completions_for_judge.py`,
`LABEL_THRESHOLD`,=0.12). Pre-selection for near-zero lexical overlap with all
references makes 0/283 weaker than an unconditioned 0/283. It remains the
relevant number for the ceiling question -- these were the only unjudged items
left, so judging them closes the split -- but it is evidence that the
remaining items were unlikely to help, not an independent replication of the
3.3% rate.

This is a testbed-validity problem, not a sample-size problem. Enlarging the
causal test's prompt pool from 467 to 750 judge-hallucinated prompts (a factor
of 1.61, not "nearly double") leaves the null intact (FFN-vs-Attention
common-site p=0.18-0.84 across all four configurations) but does not raise the
flip-rate ceiling at all, because the added prompts contributed no correct
answers to flip toward. Nor does it monotonically improve resolution: for the
same common-site comparison the minimum detectable odds ratio moves from
3.25-6.0 at n=467 to 2.5-8.0 at n=750 -- tighter at three configurations
(L8/alpha=20: 3.25to2.5; L8/alpha=40: 3.25to2.625; L9/alpha=40: 4.0to3.0) and
*looser* at the fourth (L9/alpha=20: 6.0to8.0), because that comparison's
discordant-pair count fell from 14 to 9 as the pool grew. Minimum detectable
effect depends on discordant pairs, not on n.

§5 item 1 states the general form of this check as a reusable procedure.

### 4.2 Degeneracy contamination of the flip-rate outcome metric

The second precondition for a flip-rate metric is that the baseline
"hallucinated" outputs are actually wrong *answers*. We checked this with a
criterion that needs no model of hallucination content at all: flag a
completion if some 4-, 5-, 6-, or 8-word phrase occurs 3 or more times in it
(`code/04_degeneration_check.py`). The check is deterministic, model-agnostic,
and costs nothing.

On three separately constructed pools: the original 70/30-split causal test
set (n=81) gives 42/81 = 51.9% (Wilson 95% CI [41.1%, 62.4%]) degenerate
repetition loops; the maximum supportable scaled test set (n=228, a leaner
15/85 split, `code/10_ffn_causal_patch_scaled.py`) gives 121/228 = 53.1% (CI
[46.6%, 59.4%]); and all 534 cached baseline completions, both label classes
together (`code/49_nondegenerate_subset_probe.py`), give 286/534 = 53.6%, or
138/268 = 51.5% restricted to Jaccard-hallucinated completions.

This is a stable property of GPT-2's TruthfulQA failure mode, not a
small-sample artifact, and its consequence for the causal test is direct: for
roughly half the test set, an intervention that "flips" the output to a
reference-matching string is at least as consistent with breaking a repetition
loop as with correcting a hallucinated claim. One gap in coverage: the 283
prompts added in §4.1 were never themselves degeneracy-checked, so the 53.6%
figure covers the original 534-item pool and the n=750 causal run inherits an
unmeasured degeneracy rate on its added prompts.

**Degeneration is near-balanced across label classes, which makes it worse,
not better.** Under the Jaccard label, 148/266 (55.6%) of *correct*-labeled
completions and 138/268 (51.5%) of hallucinated-labeled ones are degenerate;
under the validated judge label, 10/27 (37.0%) and 276/507 (54.4%). A
word-overlap heuristic credits repetition loops as correct about as often as
it penalizes them. So degeneration does not merely add noise to one arm of a
flip-rate comparison; it is present on both sides of the label boundary and on
both sides of the intervention comparison.

**Effect on the causal test.** Restricting the n=228 scaled test to the 107
prompts confirmed non-degenerate, the FFN-found-vs-FFN-random null holds if
anything more uniformly (McNemar p=1.000, 1.000, 1.000, 0.503 at L8/alpha=20,
L8/alpha=40, L9/alpha=20, L9/alpha=40, versus p=0.522, 0.868, 1.000, 0.659
unfiltered). Filtering does not rescue an effect.

**Effect on the passive probe.** A live alternative explanation for the
passive probe's above-chance AUROC is that it detects degeneracy-propensity --
"this prompt will send GPT-2 into a loop" -- rather than hallucination
content. We rerun `code/02`'s exact probe protocol on the non-degenerate
subset only, at every layer, for both components, under both labels
(`code/49_nondegenerate_subset_probe.py`; n=248 of 534 survive, 118
Jaccard-correct and 17 judge-correct). Because a single fold seed moves these
estimates by more than several of the effects this paper compares (§4.5), each
number below is a mean over six CV fold seeds (0-5); per-cell values at
`code/02`'s own seed 42 are in the result JSON and change no conclusion here.

**Under the Jaccard label, the passive signal survives the filter and in fact
strengthens.** FFN's peak rises from 0.6155 (L8, full pool) to 0.6305 (L8,
non-degenerate); Attention's peak from 0.6160 (L3) to 0.6812 (L3); the average
AUROC across all 12 layers rises for both components (FFN 0.5619to0.6015; Attn
0.5551to0.5923); and the FFN layer-count majority is essentially unchanged
(8/12to7/12). **So the passive probe is not merely a degeneracy detector** --
if it were, removing every degenerate completion from both classes should have
collapsed it, and instead every summary improves. Some of that improvement
plausibly reflects a cleaner label, since the Jaccard heuristic's worst errors
are exactly the repetition loops it credits as correct (§6): the filter
removes label noise, not signal.

**Under the validated judge label the same test is uninformative.** The filter
leaves only 17 judge-correct items, and at that positive count the estimates
are unstable and mostly fall: FFN's peak moves from 0.6429 (L5, full) to
0.5756 (L11, non-degenerate) and its all-layer average from 0.5757 to 0.4941
(below chance), while Attention's peak is roughly unchanged (0.6772 at L5 ->
0.6813 at L5) and its all-layer average falls from 0.6083 to 0.5442. We do not
read the FFN drop as evidence that FFN's signal specifically was
degeneracy-driven: with 17 positives across five CV folds, neither component's
estimate is powered to distinguish a real change from noise. The exclusion of
degeneracy-propensity therefore holds under one label only; under the other,
this test resolves nothing.

### 4.3 A direction-validity gate, and its exact power

Before any direction is injected, does it separate the two classes on
genuinely held-out data? The causal kernel splits a 58-prompt training pool
(18 judge-correct, 40 judge-hallucinated) into a direction-fit set (47) and a
held-out validity set (11: 3 correct, 8 hallucinated), estimates each
difference-of-means direction on the former, and measures its scalar
projection's AUROC on the latter.

**That n=11 is a default, not a data limit.** The kernel caps its training
pool at `TRAIN_N_PER_CLASS` =40 per class and takes the last 20% of each class
as the validity holdout, which gives 40/5=8 negatives. The judge-labeled pool
contains 507 hallucinated items; the kernel uses 40 and leaves 467 in the
causal test pool, and since only 32 negatives are spent fitting the direction,
475 were eligible as gate negatives all along at no data or compute cost.
*Positives* are the binding constraint -- there are 27 judge-correct items in
the whole 534-item pool, and this split spends 15 of them on direction-fitting
-- so we report the gate as a function of both class counts rather than of a
single "held-out size" (Tables 2 and 3; Appendix A item 9).

**The gate result, at both holdout sizes.** At the kernel's 3:8 holdout,
neither direction clears chance in the helpful direction at either layer: L8
FFN/Attn AUROC =0.083/0.083 (bootstrap 95% CI [0.0,0.333] both), L9 FFN =0.0
(CI [0.0,0.0]), L9 Attn =0.125 (CI [0.0,0.5]), and an exact Mann-Whitney test
at n_+=3, n_-=8 finds three of the four nominally significant in the
*anti-predictive* direction (exact two-sided p=0.0485, 0.0485, 0.0121; L9 Attn
p=0.0848 is not; under Bonferroni only L9 FFN survives, and on a consistently
two-sided convention the two L8 cells sit just under 0.05 rather than
comfortably below it, Appendix A item 5). **Re-scoring the identical direction
-- fit on exactly the same 47 items -- against all 475 eligible negatives
instead of 8 removes that anti-predictive appearance entirely**
(`code/54_enlarged_negative_holdout_gate.py`, which first reproduces the
kernel's 3:8 values exactly): the held-out AUROCs become 0.393 (L8 FFN), 0.304
(L8 Attn), 0.277 (L9 FFN) and 0.183 (L9 Attn), exact two-sided p=0.544, 0.255,
0.194 and 0.056 -- *none* significant. The directions are uninformative, not
anti-informative; the earlier anti-predictive appearance was an artifact of an
8-negative holdout the data never required.

**The single split is also an unlucky draw.** Redrawing which items land in
the fit and holdout roles at 200 random seeds on the identical 534-item
judge-labeled pool (`code/46_direction_validity_resplit_diagnostic.py`), the
held-out AUROC has mean 0.54-0.58 and SD 0.20-0.22, range [0.083,1.0] at L8
and [0.0,1.0] at L9. The kernel's single seed sits in the extreme low tail:
only 1.5% (L8 FFN), 1.5% (L8 Attn), 0.5% (L9 FFN) and 4% (L9 Attn) of resplits
land at or below it. A second, independent draw from a differently-ordered
pool (the enlarged-pool run) gives 0.375/0.333 (L8 FFN/Attn) and 0.167/0.25
(L9 FFN/Attn) -- inside the typical range. Two alternative estimators
(logistic-regression weights, Fisher LDA) on the identical split do no better:
all 12 layer/component/estimator combinations land at or below AUROC 0.167
(`code/44_alternative_direction_estimators.py`).

The resplit means are *not* significantly above chance, and the naive reading
reverses that conclusion. Treating the 200 resplits as independent draws gives
SE 0.014-0.015 and hence z=2.65-5.62 against 0.5, i.e. "the direction *does*
carry signal" at all four cells. But every resplit draws its 3 positives from
the same 27 judge-correct items, so that SE is far too small. The correct null
is a label permutation: permute the 534 judge labels (preserving the 27/507
class counts, and therefore the split structure), rerun the whole 200-resplit
procedure under each permutation, and compare the observed resplit mean
against the distribution of permuted resplit means, which inherits the same
dependence (`code/53_resplit_permutation_null.py`, 2,000 permutations). The
null is centered at chance (0.501-0.502) with SD 0.051-0.052, about 3.4-3.6x
the naive SE, and against it no cell reaches significance (Table 1). **The
3.5x SE inflation is the reusable part**: any diagnostic that resamples splits
of a fixed small pool and reports a mean over resamples has this problem.

- Cell | obs. mean | naive z | null mean | null SD | perm. z | one-sided p

- L8 FFN | 0.5787 | 5.62 | 0.5017 | 0.0509 | 1.51 | 0.085

- L8 Attn | 0.5660 | 4.45 | 0.5013 | 0.0520 | 1.25 | 0.109

- L9 FFN | 0.5525 | 3.57 | 0.5015 | 0.0515 | 0.99 | 0.164

- L9 Attn | 0.5408 | 2.65 | 0.5006 | 0.0519 | 0.78 | 0.215

*Label-permutation null for the 200-resplit direction-validity diagnostic.
"naive z" treats the 200 resplits as independent draws (SE = SD/sqrt200); it
is reported only to show what an untested reading would conclude, and is wrong
because every resplit reuses the same 27 positives. The permutation null
permutes the 534 judge labels and reruns the entire 200-resplit procedure,
2,000 times.*

**The gate's exact operating characteristics.** Under the null that a
direction carries no information, the held-out AUROC is a rescaled
Mann-Whitney U statistic whose distribution is exact and enumerable over all
binomn_++n_-n_+ label arrangements. We compute it exactly
(`code/51_direction_validity_mde_table.py`) along two axes: Table 2 grows both
classes together at this paper's 3:8 ratio, and Table 3 holds positives at 3
and varies the negatives, which is the axis this testbed could actually have
moved along. At the kernel's 3:8 holdout the null arises from only 165
equiprobable arrangements and the AUROC takes just 25 attainable values
(spacing 1/24), so the smallest *observable* AUROC that could ever be called
significant at one-sided alpha=0.05 is 0.875 (exact size 0.0424); a pure-noise
direction passes a ">= 0.75" gate 13.9% of the time and a ">= 0.80" gate 6.7%;
and power is 0.31 against a true AUROC of 0.75 and 0.72 against 0.90. **Every
one of those numbers improves at zero data cost once the available negatives
are used**: at 3:475 the MDE falls to 0.779, the exact size rises to 0.0496
(the discreteness penalty essentially disappears), the false-accept rates fall
to 7.1% and 3.7%, and power rises to 0.47 at a true 0.75 and 0.90 at a true
0.90. The one number that does not improve is the naive "AUROC >0.5" rule,
46.1% to 50.0% -- it was never a gate, and at 3:8 it only looked better
because discreteness put mass exactly at 0.5.

Most of that gain arrives with the first few dozen negatives (Table 3: n_-=60
already reaches MDE 0.789 and a 4.3% false-accept rate at the >=0.80 rule),
and the curve is flat thereafter, so the recommendation is not "use every
negative" but "do not let a per-class cap silently set your holdout."
Negatives cannot buy the rest: with positives held at 3 the MDE never falls
below 0.78 however many are added. A gate that both admits moderate true
effects and rejects noise at the 1% level needs positives -- at 10 positives
against the same 475 negatives the MDE is 0.652, the >=0.75 false-accept rate
is 0.3%, and power at a true 0.75 is 0.89 -- and this testbed has 27 correct
items total, so spending 10 on the holdout leaves 17 to fit the direction.
**The binding constraint is the positives, and a trade-off between gate power
and direction quality that 27 of them cannot satisfy simultaneously.**

- n | n_+ | arrangements | MDE | exact alpha | FA >0.5 | FA >=0.75 | pow.
  @0.75 | pow. @0.90

- 11 | 3 | 1.65x10^2 | 0.875 | 0.0424 | 0.461 | 0.139 | 0.31 | 0.72

- 18 | 5 | 8.57x10^3 | 0.769 | 0.0473 | 0.500 | 0.059 | 0.51 | 0.93

- 29 | 8 | 4.29x10^6 | 0.708 | 0.0464 | 0.490 | 0.021 | 0.69 | 0.99

- 37 | 10 | 3.48x10^8 | 0.681 | 0.0488 | 0.493 | 0.010 | 0.79 | 1.00

- 55 | 15 | 1.19x10^13 | 0.647 | 0.0493 | 0.496 | 0.002 | 0.92 | 1.00

- 73 | 20 | 4.30x10^17 | 0.626 | 0.0495 | 0.498 | 0.000 | 0.97 | 1.00

- 110 | 30 | 8.37x10^26 | 0.603 | 0.0498 | 0.499 | 0.000 | 1.00 | 1.00

*Exact operating characteristics of a held-out direction-validity gate when
*both* classes grow, at this paper's own 3:8 positive:negative ratio. "MDE" is
the smallest *observable* held-out AUROC that can be significant at one-sided
alpha=0.05 under the exact null -- a property of the statistic, not a bound on
the true effect; the power columns give the chance a given true effect clears
it. "exact alpha" is that test's true size (discreteness makes it smaller than
0.05); "false-accept" columns give the exact probability that a pure-noise
direction passes the stated gate rule; power is Monte Carlo (20,000 draws)
under a binormal alternative. The first row is this paper's own gate.*

- n_- | n | arrangements | MDE | exact alpha | FA >=0.75 | FA >=0.80 | pow.
  @0.75 | pow. @0.90

- 8 | 11 | 1.65x10^2 | 0.875 | 0.0424 | 0.139 | 0.067 | 0.31 | 0.72

- 20 | 23 | 1.77x10^3 | 0.817 | 0.0469 | 0.098 | 0.058 | 0.40 | 0.84

- 60 | 63 | 3.97x10^4 | 0.789 | 0.0499 | 0.080 | 0.043 | 0.44 | 0.88

- 200 | 203 | 1.37x10^6 | 0.782 | 0.0492 | 0.073 | 0.038 | 0.45 | 0.89

- 475 | 478 | 1.81x10^7 | 0.779 | 0.0496 | 0.071 | 0.037 | 0.47 | 0.90

*The same exact calculation with *positives held at 3* and only the negative
count varied. The first row is the gate this paper actually ran; the last is
what the same 3 positives would have bought against every negative not spent
on fitting the direction (475 of the pool's 507 judge-hallucinated items). No
new data is involved in any row. Columns as in Table 2, plus the false-accept
rate of a ">=0.80" rule.*

The conclusion is narrow and unavoidable: the directions this paper patches
were never shown to carry label-relevant signal, and at the number of
*positives* this testbed supplies they could not have been. The held-out size
itself, by contrast, was never forced by the data -- a reader auditing a gate
should check which of its inputs is a data limit and which is a default.

### 4.4 The causal-patching null, and why it is not interpretable

**Design.** A difference-of-means "truthfulness direction" computed on FFN
sublayer output over a train split only is injected additively into the FFN
sublayer during generation, at L8/L9 and alpha in 20,40, against a
random-direction control and against a genuinely Attention-derived direction
injected at the Attention site. The two directions are nearly orthogonal
(cosine -0.054 at L8, -0.056 at L9), so the component-specificity control is a
real control, not a relabeled copy of the treatment
(`kaggle_kernels/paper1-causal-patch-real-attn-direction/`).

**Results, under the validated label at n=467 judge-hallucinated prompts.**
Flip-to-correct rates sit at 1.3-3.0% in every one of the sixteen found/random
x component x layer x alpha conditions. FFN-found vs. Attention-found gives
McNemar p=0.167 (L8/alpha=20), 0.607 (L8/alpha=40), 1.000 (L9/alpha=20), 0.077
(L9/alpha=40) -- none significant, and the smallest would not survive a mild
multiple-comparison correction across four configurations. Each arm is also
indistinguishable from its own random-direction control (FFN: p>=0.52
throughout; Attention: p=0.84, 0.50, 1.00, 0.18), and rescoring the same
prompts and patches by the Jaccard heuristic gives FFN-vs-Attention p=0.439,
0.355, 0.747, 0.399 -- also uniformly non-significant.

**What this null can and cannot exclude.** Discordant pairs per configuration
are 19, 15, 9, 16; at those counts the minimum odds ratio an exact two-sided
McNemar test can detect is 3.75-8.00 -- a different quantity from the
3.25-6.00 common-site range in §4.1, which Appendix B separates out alongside
the observed odds ratios and intervals. Equivalence is not established either:
a TOST search finds no cell reaching the pre-registered OR=2.0 bound, with
smallest achievable bounds of 2.85 to 10.85 across the eight tested cells.
This test can neither detect a difference nor assert its absence.

**The random-direction ensemble, read correctly.** At the flagship
configuration (L8, alpha=40, 60 test prompts) the found directions' flip rates
were compared against 20 independently-drawn random directions. **The found
directions produced 0/60 flips each; 10 of 20 random FFN directions and 8 of
20 random Attention directions also produced 0/60.** The reported "50th
percentile" (FFN) and "40th percentile" (Attention) are therefore tie counts
at the floor of the statistic, not locations in a distribution. **This check
is uninformative at this flip-rate floor and is not evidence that the found
direction is special**, in either direction. The same applies to the
enlarged-pool rerun's "95th" and "100th" percentiles, driven by 2/60 and 3/60
flip events against an ensemble whose members mostly score 0/60 or 1/60. We
report both runs and treat neither as evidence.

**Four supporting controls, all consistent with the same null** (Appendix B).
(i) A common-injection-site test confirms that FFN-native-site and
FFN-common-site patching are mathematically identical for GPT-2's block
structure, and finds the genuinely new Attention-common-site comparison
non-significant at every configuration -- the single nominal p=0.049 does not
survive Holm correction across even its own four tests. (ii) A dosage
diagnostic indicates that the correct denominator is the residual stream
rather than each sublayer's own output norm, but it *cannot* demonstrate the
absence of a dosage asymmetry: it uses one shared denominator for both arms,
which makes their relative perturbation equal by construction rather than by
measurement. (iii) A low-dose sweep at alpha in 2.5,5,10 is flat, with no
monotone increase in either arm and the Attention arm in fact monotone
*decreasing* -- the opposite of a dose-responsive steering effect. (iv) A
differential-degeneration confound in the outcome metric itself is disclosed
and left unresolved: the found direction degenerates more than a norm-matched
random one at L8, but the ordering reverses at L9. Extending the test to
Pythia-410M and Qwen2.5-0.5B-Instruct is uninformative rather than merely
underpowered, valid pairs collapsing to zero on both at alpha=40.

**The honest reading.** Every one of these tests returns a null, and none is
informative about component specificity: §4.3 shows neither injected direction
was shown to carry label-relevant content, §4.1 shows the outcome variable is
floored by model competence, and §4.2 shows over half the outcome events that
do occur are confounded with loop-breaking. A null from an instrument that
cannot be shown to measure anything is uninterpretable between "FFN and
Attention are equally (un)causal for hallucination" and "we injected two
vectors indistinguishable from noise, so of course nothing differed." We
report the latter as not excluded, which is the same as reporting the
instrument inconclusive.

### 4.5 The peak-versus-peak comparison is smaller than fold-seed noise

On GPT-2 under the Jaccard label, FFN wins 8/12 layers (two-sided binomial
p=0.39; one-sided p=0.19; neither significant). Peak FFN layer is L8 (AUROC
0.6053); peak Attn layer is L3 (AUROC 0.6165) -- so the single
best-discriminating component on GPT-2 is Attention, not FFN.

**How large is a fold seed worth?** Two numbers this repository reports for
the *same* features (GPT-2 FFN L8) differ: 0.6053 here and 0.643 in the
paired-delta analysis of `code/37_paired_component_delta_auroc.py`. *Three*
things differ between the two scripts, and the dominant one is the
cross-validation fold seed, not the aggregation convention (mean-of-folds
versus pooled out-of-fold predictions) that would be suspected first:
`code/02_cross_arch_component_probe.py` uses
`StratifiedKFold(random_state=42)` against `code/37`'s `random_state=0`, and
`code/37` additionally fit its `StandardScaler` on the full dataset before
`cross_val_predict` rather than inside a per-fold `Pipeline`, a genuine (if
tiny) leak now fixed. Decomposing exactly
(`code/50_cv_seed_sensitivity_sweep.py`): mean-of-folds is 0.6053 at seed 42
and 0.6418 at seed 0 (seed component +0.0365); leak-free pooled-OOF at seed 0
is 0.6427 (aggregation, +0.0010); and the scaler-placement term, from
differencing `code/37`'s pre- and post-fix output at fixed seed and
aggregation, is -0.000056. **The seed accounts for 0.0365 of the 0.0374 total
gap, aggregation for 0.0010, and the scaler leak for -0.00006.** Rerunning
`code/37` leak-free changes no conclusion: every Delta, BCa interval and peak
layer here is identical at the precision reported, bar two third-decimal moves
(GPT-2's naive Attention peak 0.632to0.633, Qwen0.5B's naive FFN peak
0.563to0.562), and no interval changes whether it covers zero (Appendix A
items 7 and 12a).

**A 50-seed sweep, and what it does and does not undermine.** We swept
`StratifiedKFold`'s `random_state` over 50 values, changing nothing else in
`code/02`'s protocol, at two layers per architecture and both components at
each. For Pythia (L11/L4) and Qwen0.5B (L8/L17) those are that architecture's
own reported peak FFN and peak Attn layer; for GPT-2 they are L8/L9, the pair
the causal experiments patch, so GPT-2's reported Attn peak (L3) is not itself
in this sweep (the 12-seed full-profile sweep below does cover every layer).
Individual component AUROCs move substantially: GPT-2 L8 FFN spans [0.5868,
0.6422] (mean 0.6154, SD 0.0132); Qwen0.5B L17 Attn spans [0.4953, 0.5948]
(mean 0.5519, SD 0.0220) and Qwen0.5B L17 FFN [0.4807,0.5724] (mean 0.5219, SD
0.0174) -- straddling chance. Across all twelve architecture/layer/component
cells swept, the full range over 50 seeds is 0.055-0.100 AUROC and the SD is
0.013-0.022. A single-seed estimate therefore carries roughly a tenth of an
AUROC point of fold-assignment spread -- an order of magnitude larger than the
0.003-0.011 peak-versus-peak margins §4.6 compares, and several times the
0.019-0.028 margins elsewhere in this paper.

**The paired same-layer difference, by contrast, is stable.** Because both
components are fit on the same folds, their difference cancels most of the
fold-assignment noise: at GPT-2 L8, Delta=+0.0688+/-0.0155 with FFN ahead in
50/50 seeds; at Pythia L11, +0.0512+/-0.0209, FFN ahead in 98%; at Qwen0.5B
L8, +0.0118+/-0.0204, FFN ahead in 70%. The sign is layer-dependent, not
architecture-dependent: at Pythia L4, Delta=-0.0569+/-0.0212 with FFN ahead in
0% of seeds, and at Qwen0.5B L17, -0.0300+/-0.0231, FFN ahead in 10%. **So the
FFN-vs-Attention answer is determined by which layer is compared, and the
peak-versus-peak comparison this literature (and this paper's earlier draft)
relies on is precisely the fragile way to choose it**, because argmax over
near-tied layers is the statistic most exposed to fold-seed noise.

**A paired test of the actual estimand.** The peak-versus-peak comparisons
above are visual: two separately-estimated CV means checked against their own
fold SDs, even though the FFN and Attention probes are fit on the same samples
and folds. The real estimand is their difference, Delta=AUROC_FFN-AUROC_Attn.
Using already-cached raw per-sample features (Pythia, Qwen0.5B) and GPT-2's
vendored mech-int activations (no new model inference), we compute out-of-fold
predicted probabilities for both components from the same folds and a BCa
bootstrap 95% CI on Delta over 2000 resamples (`code/37`). At each
architecture's own FFN-peak layer under this aggregation: GPT-2 (L8)
Delta=+0.067, CI [+0.012,+0.122]; Pythia (L11) Delta=+0.047, CI
[-0.002,+0.100]; Qwen0.5B (L20) Delta=+0.053, CI [-0.007,+0.113]. At each
architecture's own Attn-peak layer the sign reverses as expected: GPT-2 (L3)
Delta=-0.085, CI [-0.137,-0.036]; Pythia (L4) Delta=-0.113, CI
[-0.162,-0.064]; Qwen0.5B (L8) Delta=-0.032, CI [-0.084,+0.023].

**These do not all agree with the 50-seed sweep above, and the disagreement is
the same effect again.** Of the two cells measured by both procedures, they
agree at GPT-2 L8 (+0.067 versus +0.0688) and Pythia L11 (+0.047 versus
+0.0512) and disagree at the other two: Pythia L4 gives -0.113 here against a
sweep mean of -0.0569, and Qwen0.5B L8 gives -0.032 against +0.0118 -- a sign
flip. The sweep is a mean over 50 fold seeds at `code/02`'s aggregation;
`code/37` is one seed at pooled-OOF aggregation. So the "stable" claim for the
paired difference is within-protocol: at a fixed aggregation, Delta varies
little across fold seeds (SD 0.016-0.023). It is not a claim that two
different protocols agree on Delta at a given layer, and at two of four shared
cells they do not.

Averaged across *all* layers rather than only the peaks -- the summary least
vulnerable to selection bias -- Delta is small on every architecture (GPT-2
+0.0017+/-0.036; Pythia +0.0005+/-0.042; Qwen0.5B -0.0021+/-0.034), and a
layer-weighted pooled estimate across all three architectures gives
Delta=-0.0003 with between-architecture variance of 3.9x10^-6. That is the
properly pooled null.

**Peak layers are not stable.** `code/37`'s pooled-OOF argmax puts Qwen0.5B's
FFN peak at L20 and its Attn peak at L8, the reverse of the mean-of-folds
peaks reported elsewhere here (FFN L8, Attn L17) -- and this is not the two
aggregation conventions disagreeing, since aggregation is worth 0.001 AUROC
against the fold seed's 0.037 on the same data, the two analyses also differ
in fold seed, and argmax over 24 near-tied layers is exactly the statistic
most sensitive to it. Recomputing the full per-layer AUROC profile at 12 fold
seeds and recording the argmax (`code/50`): on Qwen0.5B the FFN peak lands on
**four** different layers (L12 in 6, L8 in 3, L20 in 2, L2 in 1) and the
Attention peak on **six** (L5 in 6, L17 in 2, and L7, L8, L12 and L23 once
each), so both "colliding" assignments -- FFN L8 / Attn L17 from `code/02`,
FFN L20 / Attn L8 from `code/37` -- are draws from that distribution. Pythia
is similar (FFN peak on 3 distinct layers, modal L11 in 10 of 12 seeds;
Attention peak on 5, with no modal layer -- L4 and L19 tie at 4 each, then L2
in 2, L9 and L17 once each). GPT-2, with half as many layers, is stable for
FFN (L8 in 12/12) but not for Attention, which ties (L3 in 6, L6 in 6).
**There is no estimator disagreement to explain: "the peak layer" is not a
well-identified quantity for either component on the 24-layer models at this
sample size.** Any claim that depends on a specific peak layer -- including
the peak-depth-fraction comparison in §4.6 -- inherits that instability.

We also checked "peak AUROC" itself for winner's-curse selection bias using
nested cross-validation (select the peak layer on inner folds only, evaluate
on a held-out outer fold). Nested-CV peaks are uniformly lower than naive
argmax peaks -- GPT-2 FFN 0.643to0.581, GPT-2 Attn 0.633to0.600; Pythia FFN
0.632to0.584, Pythia Attn 0.666to0.582; Qwen0.5B FFN 0.562to0.521, Qwen0.5B
Attn 0.570to0.515. On GPT-2 the correction flips which component leads (naive:
FFN 0.643 vs. Attn 0.633; nested: Attn 0.600 vs. FFN 0.581). We do not read
this as Attention "really" winning -- the corrected margin is itself within
noise -- but as a concrete demonstration that peak-AUROC comparisons can
select the noisier rather than the truer component.

**Three sign flips out of three.** The "naive argmax" column above disagrees
with §4.6's peak table on *which component wins, on all three architectures*
-- here FFN leads on GPT-2 and Attention on Pythia and Qwen0.5B, while §4.6,
computing the same quantity at `code/02`'s fold seed and aggregation, has the
opposite winner in each case. Both sets are correct for their own protocol.
Three sign flips out of three, produced by nothing but a fold seed, an
aggregation convention and a scaler placement worth 0.00006, is the strongest
single piece of evidence in this paper that the peak-versus-peak
FFN-vs-Attention comparison is not measuring anything at this sample size. We
report §4.6's version only because it is the protocol the rest of the passive
results use, not because it is the right answer.

### 4.6 Cross-architecture passive results

[Real Kaggle data, N=605 Pythia / N=513 Qwen0.5B; per-layer numbers in
`results/cross_arch_component_probe_*.json`.] Under the Jaccard label and a
bare prompt template, FFN wins a numerical majority of layers on all three
architectures (66.7%, 66.7%, 58.3%); per-architecture two-sided p-values are
0.39/0.15/0.54, one-sided 0.19/0.076/0.27, all non-significant. Pooled across
all 60 layers, 38/60 FFN wins gives a nominally significant one-sided p=0.026,
but this is not a valid inferential instrument: 60 layers within only 3
architectures are strongly autocorrelated, not independent trials. The only
cleanly poolable count is architecture-level (3/3), directionally consistent
but far too small an n to test formally.

On peak AUROC, Attention is the single best-discriminating component on
**one** of three architectures (GPT-2). Pythia's peak favors FFN (L11 =0.6181
vs. Attn L4 =0.6115); Qwen0.5B's also favors FFN (L8 =0.5657 vs. Attn L17
=0.5625). Every one of these margins is a small fraction of the relevant
peak's own cross-validation SD: GPT-2's 0.011 margin against SDs of
0.0557/0.0427 (about a quarter to a fifth), Pythia's 0.0066 against
0.0442/0.0345 (about a seventh to a fifth), and Qwen0.5B-bare's 0.0032 against
0.0628/0.0423 (about a twentieth to a thirteenth) -- and, per §4.5, all of
them are a small fraction of what changing the fold seed alone is worth. **The
uniform statement is: the peak-component question is within measurement noise
on all three architectures tested.**

**A template confound on Qwen0.5B.** Qwen2.5-0.5B ( 494M parameters, the
largest model tested) was queried with a bare `Q: ... A:` template rather than
its chat template -- genuinely out-of-distribution usage for an
instruction-tuned model. Rerunning with the proper chat template
(`code/02_cross_arch_component_probe.py qwen05chat`) reverses the
peak-component result: Attention becomes the peak (L4 =0.5988+/-0.0438) over
FFN (L4 =0.5704+/-0.0186), and the layer-count result reverses too (FFN wins
11/24, 45.8%, versus 58.3% before). The corrected margin (0.0284) is still
smaller than Attention's own CV SD (0.0438), so "Attention now clearly wins"
would overclaim in the same way "FFN wins 3/3" did. The finding is the
direction of the flip, not a newly-resolved winner. Peak-FFN depth fraction
also shifts, from 33.3% (L8/24, bare) to 16.7% (L4/24, chat).

**But more than the prompt changed, and the reversal does not survive
relabeling.** The two Qwen conditions are also different samples (N=513 bare
versus N=433 chat, with only 298 questions common to both), because the
Jaccard filter that admits an item depends on the completion, which the
template changes. That composition difference is of the same order as the
0.0284 margin the reversal turns on, so the template result is confounded with
sample composition rather than a clean single-variable manipulation (Appendix
C). The stronger reason not to lean on it is relabeling. Rescoring every
completion on all three architectures with the LLM judge raises absolute
AUROCs substantially everywhere (GPT-2 0.605/0.616to0.698/0.717 FFN/Attn peak,
and similarly on the others) and restores FFN's numerical majority on
Qwen0.5B-chat (11/24to18/24). Under the validated label FFN's majority *rises*
on three of four architecture/template conditions and only GPT-2 moves the
other way; and only 2 of 4 conditions keep the same peak winner under both
labels. Which component leads is label-sensitive at every level at which we
measured it.

**A class-imbalance caveat on every validated-label number above.** The judge
label is severely imbalanced toward "hallucinated": GPT-2 27/534 correct
(5.1%), Pythia 29/605 (4.8%), Qwen0.5B-bare 63/513 (12.3%), Qwen0.5B-chat
73/433 (16.9%). Cross-validated AUROC at this imbalance is substantially
noisier: on GPT-2 the FFN/Attn peak AUROC standard deviations roughly double
under the validated label (FFN 0.0557to0.1115; Attn 0.0427to0.1253), so
GPT-2's own 0.698/0.717 margin (0.019) is well within one SD of either peak.
These shifts are real properties of the re-analysis, and suggest an
under-characterized effect of label quality on this probe -- not tighter
estimates than the numbers they revise.

![figure](figures/ffn-attn-comparison.pdf)

*Peak AUROC for FFN vs. Attention across every tested condition, with error
bars showing each peak's own cross-validation standard deviation. The margin
between components is within one CV SD of overlap in all four conditions, and
Qwen0.5B -- whose narrow bare-template FFN edge (0.0032, the smallest of the
three) is the one that reverses -- appears under both templates, favoring
Attention once queried with its proper chat template. §4.5 shows that changing
only the cross-validation fold seed moves these estimates by more than the
margins shown.*

### 4.7 Further instruments and controls

Three further lines of evidence are consistent with everything above without
changing it. Each is reported in full in an appendix; we state the headline
here.

**Two further causal instruments** (Appendix D). Additive mean-shift steering
is a comparatively weak causal instrument, so we ran two stronger ones and
both return nulls. ROME-style causal tracing (Meng et al. 2022), adapted to
closed-book QA by corrupting the whole question span, finds no (layer,
component) cell surviving a joint Holm-Bonferroni correction on any of the
three architectures under either label; its judge-label rerun runs into §4.1's
competence ceiling again, with only 17 usable candidates among the 27
judge-correct GPT-2 samples. A sparse-feature intervention fails one stage
earlier still: substituting a layer-8 SAE (d_sae=24,576) for the dense
direction, 0 features survive Benjamini-Hochberg FDR at q=0.05 on this paper's
own dataset, so the causal clamp step was never reached. An identically-run
positive control on a companion dataset finds 331 surviving features, which is
evidence that the null is not a feature-selection bug, though it leaves it
bounded by a disclosed instrument mismatch.

**Difficulty-matched controls, which had no power** (Appendix E). The
FFN/Attention signal survives matching correct and hallucinated groups on two
question-difficulty proxies essentially unchanged. That is weaker evidence
than it looks: *neither proxy correlates significantly with correctness before
matching* (entropy r=0.045, p=0.295; composite r=0.024, p=0.578), so this
dataset never had a statistically detectable difficulty confound for the
control to remove. It is survival under a weak test, not dissociation under a
strong one. Under the validated label the same control is uninformative,
leaving 54/534 samples after matching, and the two proxies then disagree. We
therefore treat the difficulty-dissociation result as established only under
the Jaccard label, and as an instrument that removed nothing.

**Layer localization is argmax over noise** (Appendix F). Six methods on the
vendored GPT-2 artifacts converge on layers 8--9 under the Jaccard label, but
three of the six are argmax over a statistic with no resolution. The dense
probe's L9 peak beats L12 by 4.5x10^-5 AUROC -- about a thousandth of that
layer's own cross-validation standard deviation (0.0497). Steering's "peak
improvement" at L9 is 0.0015 AUROC, the argmax over a 52-cell grid in which
the improvement is exactly zero at 11 of 13 layers. And the logit-lens
analysis yields two divergence-layer estimates from one run, only one of which
is L8. **The convergence does not survive relabeling**: rerunning the four
activation-only methods under the judge label moves the peaks to L7, L7, L6
and L11. Layers 8--9 are a property of the Jaccard label's noise pattern on
this dataset, not a label-independent localization.

### 4.8 A leave-one-category-out result whose interpretation changed twice

We report this subsection in full, including the interpretations we abandoned,
because the sequence of checks -- not the final number alone -- is the
transferable result. An earlier draft reported its first half as a headline
finding; an independent review then argued the whole thing was an estimator
artifact; the controls below support neither reading exactly.

**The diagnostic, and its first reading.** TruthfulQA questions cluster into
38 topical categories, and standard random K-fold CV -- the protocol every
passive probe number here uses -- can place same-category questions in both a
fold's train and test split. Category is not independent of the label: on this
534-item GPT-2 pool, among the 10 most frequent categories (each n>=15) the
judge-correct rate ranges from 0% (Fiction, Paranormal, Stereotypes) to 10.5%
(History, Conspiracies), and across all 38 from 0% to 28.6% (Confusion:
People, n=7). We ran a leave-one-category-out (LOGO) CV re-test for the
component probe at GPT-2 L8/L9 (`code/47_category_leakage_diagnostic.py`):
train on 37 categories, test on the held-out one, repeated for every category
with both classes present (16 of 38 qualify; the other 22 have zero
correct-labeled items). Standard 5-fold CV gives AUROC 0.622 (L8 FFN), 0.632
(L8 Attn), 0.616 (L9 FFN), 0.663 (L9 Attn); LOGO-CV gives 0.479, 0.491, 0.482,
0.489 (SD 0.27-0.35 across folds). The first reading was: category-clustering
leakage explains the probe's signal.

**The objection: is the "collapse" just the LOGO estimator's own variance?**
The objection has real force. The LOGO implementation averages *per-category*
AUROCs, a different estimand from the per-fold average over random folds that
standard CV reports. With 22 of 38 categories having zero positives and the
surviving 16 averaging 1.7 positives each, each fold's AUROC is close to a
coin flip. A naive standard-CV-versus-LOGO comparison is also not significant
on its own terms (Welch p=0.09-0.16 across the four cells) -- but it compares
two different estimands, so its p-value does not answer the question either.

**Two permutation controls settle whether the estimator is to blame.** We
rerun the identical LOGO procedure (`code/47::probe_leave_one_category_out`,
reproduced unmodified) on group assignments random by construction, 1,000
draws each (`code/48_permuted_pseudocategory_control.py`): *size-matched*,
permuting the category-assignment vector so group sizes are preserved exactly;
and *size- and class-matched*, permuting positives among positive slots and
negatives among negative slots so every pseudo-category has exactly the same n
*and* the same number of correct-labeled items as the real category it
replaces. The latter makes the usable-fold count identical to the real one
(16, versus 17.34 on average under size-matching alone) and removes the
alternative explanation that the 16 usable real categories are simply an
unusual subset. **Under both controls, LOGO on pseudo-categories recovers
approximately the standard-CV AUROC rather than chance:**

- Cell | standard 5-fold CV | real LOGO | size-matched null |
  size+class-matched null

- (p; Holm) | (p; Holm)

- L8 FFN | 0.6215 | 0.4788 | 0.6163+/-0.0410 (0.006; 0.018) | 0.6153+/-0.0402
  (0.004; 0.008)

- L8 Attn | 0.6318 | 0.4907 | 0.5820+/-0.0412 (0.040; 0.040) | 0.5816+/-0.0408
  (0.022; 0.022)

- L9 FFN | 0.6157 | 0.4816 | 0.6140+/-0.0395 (0.002; 0.008) | 0.6121+/-0.0406
  (0.002; 0.008)

- L9 Attn | 0.6632 | 0.4891 | 0.6167+/-0.0413 (0.006; 0.018) | 0.6192+/-0.0404
  (0.002; 0.008)

*Leave-one-category-out AUROC on real TruthfulQA categories versus two
permuted-pseudo-category nulls (1,000 draws each), GPT-2, judge label.
Empirical two-sided p is computed against the corresponding null; 0.002 is the
attainable floor at 1,000 permutations. "Holm" is the Holm-Bonferroni adjusted
p across the four cells within each control family, the same correction used
in §4.4 and Appendix D. An earlier version ran 100 permutations, whose p floor
(0.0198) is coarser than the smallest Holm threshold (0.0125), so no cell
could have survived correction at any effect size (Appendix A item 8e).*

The per-category-averaging estimand is therefore *not* biased downward at this
sample size: on random groupings of identical size and class composition it
returns 0.58-0.62, within noise of the standard-CV values. The real-category
LOGO value sits below that null at every cell and outside it at all four: 1,
10, 0 and 0 of 1,000 size- and class-matched draws fall at or below the real
value (L8 FFN, L8 Attn, L9 FFN, L9 Attn), and all four survive Holm-Bonferroni
correction under both controls (adjusted p=0.008-0.040). **The collapse is
specific to real topic structure, not an artifact of the estimator** -- an
outcome we had predicted would go the other way.

**A second thing LOGO changes, which has to be separated out.**
Leave-one-category-out alters the protocol in two ways at once, and only one
of them is about leakage. (i) It changes which pairs enter the AUROC: a
per-category AUROC compares a positive and a negative drawn from the *same*
category, whereas standard K-fold CV pools all pairs, of which only 4.9% (675
of 13,689) are within-category on this pool. (ii) It removes same-category
items from the training split. We separate these by decomposing the *standard*
CV's own out-of-fold scores into within- and between-category pairs
(`code/48`); this holds the training protocol fixed and varies only the pair
set.

Separating them requires holding the *estimand* fixed too. A pooled
within-topic AUROC over all 675 within-topic (positive, negative) pairs is
pair-weighted -- a 69-item category contributes many more pairs than a 4-item
one -- whereas the LOGO column is an unweighted mean of 16 per-category
AUROCs. Subtracting the second from the first silently swaps one weighting
scheme for another instead of isolating the effect of removing same-topic
training data. We therefore recompute the within-topic column a second way
(`code/52_estimand_matched_within_topic_auroc.py`): the mean of per-category
AUROCs from the *standard*-CV out-of-fold scores, over exactly the 16
categories LOGO uses. That quantity differs from the LOGO column in one
respect only -- whether same-topic items were present in training -- so its
difference from LOGO is a decomposition rather than an artifact.

**Held at a fixed estimand, the training-overlap residual is indistinguishable
from zero.** It is -0.011, +0.023, -0.041 and +0.061 across the four cells --
negative in two of them -- and no cell is close to significant (paired
p=0.15-0.85). Removing same-topic items from the probe's training data does
not measurably degrade its within-topic discrimination, because that
discrimination was already at chance with them present: the estimand-matched
within-topic AUROC is 0.441-0.550. *This probe's within-topic discrimination
is at chance regardless of whether same-topic items are in its training data.*
There was no cross-topic signal to lose. (An earlier version subtracted the
pair-weighted within-topic column, 0.536-0.684, from the LOGO column and
reported a 0.06-0.20 AUROC "training overlap" effect; that entire quantity was
the mechanical difference between two weighting conventions. Appendix A item
8d.)

What the pair decomposition does establish is where the pooled AUROC's
discrimination lives: essentially all of it is carried by between-topic pairs
(0.609-0.648, on 95.1% of the pairs), not within-topic ones -- which is
consistent with, though it does not establish, the topic-correlated-feature
account below.

**But we cannot name the mechanism.** The originally claimed mechanism -- the
probe reading off each topic's correct-answer rate -- makes a checkable
prediction: a classifier given *only* the category label should recover much
of the 0.62-0.66 AUROC. On the identical pool and labels (`code/48`), an
in-sample per-category correct-rate rule reaches AUROC 0.7938, but
leave-one-out gives 0.5054 and this paper's own 5-fold CV protocol gives
0.4863 (pooled-OOF) and 0.4789 (mean-of-folds). **Estimated the way any real
probe would have to estimate it, topic identity alone is at or below chance.**
That does not exclude a topic-base-rate account on effect-size grounds -- the
in-sample 0.7938 shows rates known exactly would suffice to exceed the probe's
own AUROC -- it shows that with 27 positives over 38 categories those rates
cannot be estimated accurately enough to test the account here. No experiment
we ran separates the competing accounts: per-topic base rates; *continuous*
topic-correlated features, which a 768-dimensional probe fit on 427 items can
exploit far more efficiently than a 38-level lookup table estimated from 1.7
positives per level, and which the ceiling calculation is *least* able to
exclude; within-topic near-duplicate and paraphrase structure; and
topic-specific feature geometry that does not transfer.

**What survives, stated narrowly.** Both readings we started from overreach.
The evidence supports this: *a standard random K-fold AUROC for this probe on
TruthfulQA overstates its performance on an unseen topic, by roughly 0.13-0.17
AUROC, and within-topic performance at the two layers tested is at chance.*
The drop is specific to real topic structure rather than to the estimator
(permutation controls); at a fixed estimand it is attributable entirely to
which pairs are discriminated, and not at all within our resolution to whether
same-topic items were in training (Table 6); and the feature-level mechanism
is unidentified. Consequently every passive AUROC in §4.5, §4.6 and Appendix F
is a *within-distribution* number including cross-topic comparisons, not an
estimate of new-topic performance. It does not differentially favor either
component (FFN and Attention drop by 0.14, 0.14, 0.13 and 0.17 AUROC), so the
FFN-vs-Attention comparison is unchanged, and no causal result in §4.3-§4.4 is
affected, those using no K-fold CV. It does mean the passive signature defined
in §3 was never shown in a form that would transfer to a new topic.

**The transferable lesson**, stated as a reusable procedure in §5 item 10: a
grouped-CV "collapse" is ambiguous between group leakage, a change of estimand
and failure to generalize across groups, and all three are cheap to separate.
We ran none of the three checks before first making the claim; each changed
our conclusion, and the last changed it twice.

## 5. A Validity Checklist for Intervention Studies

This is the constructive contribution, written to be lifted out and used
independently of anything about FFNs. Items 1--8 apply to any study that
estimates a direction (or feature, or activation) from labeled data, patches
it into a model, measures an outcome rate, and draws a causal conclusion;
items 9--10 apply to passive probing. Each is one this paper's own flagship
experiment either failed or needed in order to be read correctly.

-  **Is the outcome metric floored by the base model's competence?** Measure
   the unassisted correct rate on the exact evaluation set with the exact
   scorer that will score the intervention, and report it. If it is near zero,
   a flip-to-correct rate cannot separate mechanisms, and more data of the
   same kind will not help (§4.1: 27/817 = 3.3%; 0 of 283 added items
   correct).

-  **Are the baseline "failures" the failure mode you think they are?** Run a
   degeneracy pre-filter before trusting a flip-rate. Ours is a single
   deterministic check -- a 4-, 5-, 6- or 8-word phrase repeated 3+ times --
   and it flags 51.9% and 53.1% of the nominally hallucinated completions in
   our two causal-test pools and 53.6% of all baseline completions in the full
   labeled pool, near-balanced across label classes (§4.2). Report the rate on
   both arms and both label classes: a filter that removes items from only one
   arm creates its own confound.

-  **Is the estimated direction validated on held-out data before any
   intervention, with a stated minimum detectable effect?** Do not report "n
   is small." The exact null distribution of a held-out AUROC is enumerable
   over all binomn_++n_-n_+ label arrangements; report the resulting MDE, the
   attainable alpha, and the false-accept rate of the gate rule actually used,
   *as a function of both class counts* (Tables 2 and 3). At this paper's own
   3:8 holdout the gate cannot call any *observed* AUROC below 0.875
   significant (true effects below it clear the gate, just rarely -- power
   0.31 at a true AUROC of 0.75), and a pure-noise direction passes plausible
   gates 6.7%-46.1% of the time. Check which class is scarce: our 8 negatives
   were a per-class training cap, not a data limit, and the 475 unused ones
   would have moved the MDE to 0.779 and halved the false-accept rates for
   free (§4.3). A patching result whose direction has not cleared such a gate
   is uninformative either way.

-  **Is the "control" direction genuinely from a different source?** A control
   that is the treatment direction relabeled, or injected at a different site,
   cannot establish component specificity; this paper's earlier version made
   exactly that mistake (Appendix A). The corrected control is a
   difference-of-means direction estimated from the other component's own
   activations, verified near-orthogonal to the treatment (cosine
   -0.054/-0.056) before use (§4.4).

-  **Does the injection site confound "which representation" with "where in
   the computation"?** Patching two sublayers at different points in a block
   entangles component identity with injection site. A common-site test
   disentangles them, and can reveal that two apparently different
   interventions are mathematically identical, as FFN-site and common-site
   patching are for GPT-2's block structure (verified by direct algebra and a
   byte-for-byte empirical check, §4.4). Relatedly, a dosage diagnostic that
   normalizes both arms by a single shared denominator finds "equal dosage"
   true by construction, and should say so (§4.4).

-  **Is a null distinguishable from a genuinely random direction -- and is the
   ensemble check even informative at the observed rate?** A found direction
   whose effect sits inside the empirical distribution of random-direction
   effects has not been shown to do anything a random direction would not. But
   read the percentile correctly: when the found direction scores 0/60 and
   half the random ensemble also scores 0/60, the "50th percentile" is a tie
   count at the floor and carries no information (§4.4). Report raw counts
   alongside any percentile.

-  **Is a null reported at the resolution the claim needs, and is equivalence
   tested rather than assumed?** Report the minimum detectable odds ratio (or
   effect size) from the discordant-pair count, not from n; the two can move
   in opposite directions as data grows (§4.1: 6.0to8.0 at one cell as n went
   467to750). If the claim is "no difference," run an equivalence test against
   a pre-specified bound and report the smallest bound achievable (§4.4: no
   cell reaches OR=2.0; achievable bounds 2.85-10.85).

-  **Does an alternative direction-estimation method change the conclusion?**
   A single estimator's failure to validate is ambiguous between "poor
   estimator" and "no effect"; checking at least one alternative distinguishes
   them (§4.3 checks logistic-regression weights and Fisher LDA).

-  **Report a fold-seed sensitivity band alongside any cross-validated AUROC
   whose claimed effect is of comparable magnitude.** On this data the fold
   seed alone moves a component AUROC over a range of 0.055-0.100 (SD
   0.013-0.022 over 50 seeds) while the margins being compared are 0.003-0.011
   (§4.5). Argmax peak-layer stability varies by architecture and must be
   checked, not assumed: over twelve seeds, GPT-2's FFN peak is perfectly
   stable (12/12 at L8) while its Attention peak splits two ways, Pythia's
   peaks take 3 and 5 distinct values, and Qwen0.5B's take 4 and 6. A claim
   that rests on which layer peaks needs a seed sweep before it can be made at
   all. A paired same-layer difference is much more stable and is the better
   estimand where the question allows it.

-  **If a grouped-CV diagnostic appears to collapse a probe's AUROC, run three
   checks before interpreting it.** A collapse is ambiguous between group
   leakage, a change of estimand, and failure to generalize across groups. (a)
   A permuted-group control, matched on both group size and per-group class
   composition, separates estimator behavior from real group structure. (b)
   Decompose the *standard* CV's own out-of-fold scores into within- and
   between-group pairs: grouped CV silently restricts the AUROC to
   within-group pairs, only 4.9% of the pairs the pooled estimate is built
   from on our data, so part of any "collapse" is a change of estimand rather
   than of performance. If you then *subtract* that within-group number from
   the grouped-CV number to isolate the training-overlap component, first make
   the two *weighting conventions* match -- a pair-weighted pooled AUROC and a
   mean of per-group AUROCs differed on our data by 0.06-0.20 AUROC, more than
   the entire effect being decomposed, and produced a headline claim we had to
   retract (§4.8, Appendix A). (c) A group-variable-only ceiling under the
   same CV protocol tests whether the group variable could carry the claimed
   effect at all; report both its in-sample and its cross-validated value,
   since the gap between them bounds what the check can conclude. These three
   moved our conclusion twice and left a claim narrower than either starting
   point (§4.8).

### 5.1 When this checklist binds, and when it does not

The checklist is not free: items 1--3 and 9 each cost a run, and item 10 costs
three. Table 5 states when this paper's own findings support paying that cost
and when they do not. Every entry is scoped to what we measured; "does not
bind" means we supply no evidence the check would change the conclusion, not
that the situation is safe.

- **Use it if:**

- **You patch an estimated direction/feature/activation and read a flip-rate
  or behavioral-change metric as evidence for a mechanism.** -> Items 1--8.
  All three headline checks fired here, leaving a null that is uninterpretable
  rather than informative (§4.4).

- **Your metric is "did the output become correct" and you have not measured
  the base model's unassisted rate with the same scorer.** -> Item 1. Ours was
  3.3% and 283 further items added zero, flooring the metric by construction
  (§4.1).

- **Your positive class is small enough that a held-out validity split is
  non-trivial -- tens of positives, not hundreds.** -> Item 3. At 27 positives
  the gate could not both admit moderate true effects and reject noise; Tables
  2--3 give the exact number for any (n_+,n_-) (§4.3).

- **Your holdout size came from a default, per-class cap or fixed split
  fraction rather than from the data.** -> Item 3. Our 8 negatives were a cap;
  using the 475 available moved the MDE 0.875to0.779 and removed an apparent
  anti-predictive result, free (§4.3).

- **You want to tell a genuine null from an uninformative one before
  publishing it.** -> Items 3, 6, 7. An ensemble percentile carries no
  information when both arms sit at the floor, and a null needs its minimum
  detectable effect and an equivalence test to be read at all (§4.4).

- **Your effect is comparable to or smaller than a fold-seed band you have not
  measured.** -> Item 9. Fold seed alone moved AUROC by 0.055-0.100 against
  margins of 0.003-0.011, giving three sign flips out of three (§4.5).

- **A grouped-CV diagnostic appears to collapse your probe.** -> Item 10.
  Three cheap checks separated leakage, estimand change and cross-group
  failure, and changed our conclusion twice (§4.8).

- **It does not bind, or needs adaptation, if:**

- **Your testbed is large and well-powered: base-model correct rate well away
  from zero, hundreds of positives.** -> Items 1 and 3 stay cheap but are
  unlikely to bind. We tested only the small-model, near-floor regime.

- **Your baseline failures are known not to be degenerate (short-form or
  constrained decoding, or a characterized generator).** -> Item 2. Our
  51.9%-53.6% rates are a property of GPT-2's greedy 40-token TruthfulQA
  completions, not a constant; run the check once to establish your regime.

- **Your intervention is already validated by an independent replication on
  the same estimand in a different setting.** -> Items 3 and 8 address a
  single unreplicated estimate; we did not test what a replication is worth.

- **Your claim is passive detection performance, with no intervention
  reported.** -> Only items 9--10 apply; the audits in §2 cover the passive
  benchmark case more directly.

- **Your effect is many times larger than your measured fold-seed band and
  discordant-pair resolution.** -> Item 9 becomes a reporting requirement, not
  a threat. We observed the opposite regime and cannot say how large a margin
  must be.

*When the checklist applies. Each row pairs a property of the study being run
with the item that bears on it and this paper's result for that item.*

## 6. Discussion and Limitations

**What this paper does and does not claim.** We claim neither that FFN
over-retrieval is present in closed-book generation nor that it is absent.
Each of our three instruments is compromised in a way we can name: the passive
probe's margins are smaller than its fold-seed noise, the causal patch's
direction cannot be validated at the available held-out size, and its outcome
metric is floored by model competence and contaminated by degeneration. We
claim only that this testbed and these instruments cannot answer the question,
and give the checks that would have to pass first.

**Scoped out: a RAG positive control.** The single most valuable missing
experiment is a positive control -- reproducing ReDeEP's original asymmetry on
a retrieval-augmented dataset with this exact pipeline, to show the pipeline
*can* detect the effect where it is reported to exist. Without it we cannot
distinguish "the effect is absent in the closed-book setting" from "this
pipeline could not detect the effect anywhere," and that limitation would have
to be closed before any mechanistic claim in either direction.

**Also scoped out.** Enlarging GPT-2's judged pool beyond the full 817-item
validation split could further probe the competence-ceiling finding, though
§4.1's 0/283 result is direct evidence that more data of the same kind would
not rescue it. We did not run the grouped-CV or fold-seed checks for the other
five layer-localization methods in Appendix F; given §4.8's conclusion those
checks are now a higher priority than we previously judged, and both are
cheap. We did not test whether §4.8's cross-topic failure is driven by
within-topic near-duplicates or by non-transferring topic-specific geometry,
which would need a paraphrase-controlled split we did not construct. We make
no comparison against black-box detectors (semantic entropy, SelfCheckGPT),
which are the production-relevant baseline, and we do not address benchmark
contamination.

**Label validity.** All original results rest on a Jaccard word-overlap label
-- surface-form divergence rather than verified factual incorrectness. We
quantified this on all three architectures with an independent LLM judge
(Qwen2.5-3B-Instruct): a 100-item stratified GPT-2 sample gives 52% raw
agreement, Cohen's kappa=0.04; full relabeling of every completion gives
kappa=0.032 (Pythia), 0.141 (Qwen0.5B-bare), and 0.084 (Qwen0.5B-chat) -- next
to chance throughout. Extending the relabel to all 534 GPT-2 completions gives
kappa=0.0417. The disagreement is consistently one-directional: the judge
calls far more completions hallucinated than the heuristic does (on the full
534-item GPT-2 relabel the judge agrees with 97.0% of Jaccard's hallucinated
calls but only 7.1% of its correct calls, computed directly from
`results/gpt2_full_534_judge_labels.json`), and manual reading of disagreement
cases indicates the judge is usually right -- word overlap frequently credits
completions that share surface words with the reference but state a different
specific fact, or that are degenerate repetition loops, as "correct" (§4.2
quantifies the latter at 55.6% of Jaccard-correct completions).

A trivial length/lexical baseline check
(`code/32_surface_baseline_vs_judge_label.py`, the same 6-feature surface
classifier rerun against the validated label on the same cached features)
gives logistic-regression CV AUROC =0.5604+/-0.1047 and MLP =0.5704+/-0.0971
(n=534, 5-fold) -- not chance-level, and not meaningfully different from the
same baseline against the Jaccard label (0.531/0.576), so the validated label
is not obviously *easier* for a surface-only classifier. That is weak evidence
against the alternative that the validated-label rise in hidden-state probe
AUROC (§4.6, roughly 0.60to0.70) is driven purely by degeneracy detection,
which would predict the surface baseline should rise too. It is not a strong
exclusion -- both baselines remain well short of the hidden-state probes --
and §4.2's non-degenerate re-probe is the more direct test. One residual
limitation: the same judge model both defines the found-direction's train
split and scores every generated output in §4.4's validated-label test, so
that result rests on the judge's own accuracy, checked here only by manual
spot-reading and the surface-feature control, not by an independent second
judge or human annotation.

**Cheap baselines.** Before attributing discriminative power to the
FFN/Attention decomposition specifically, we checked whether trivial
generation-confidence signals do comparably well
(`code/41_cheap_baselines.py`, same n=534 validated label): an undecomposed
last-layer probe reaches AUROC =0.610+/-0.126; teacher-forced
generation-confidence features individually reach 0.54-0.64, with
min-max-softmax strongest at 0.643+/-0.065, and all four combined at
0.635+/-0.077. These sit in the same range as this paper's FFN/Attention
component probes (0.53-0.75 depending on layer and label), so the specific
decomposition studied here does not obviously outperform a mechanism-agnostic
confidence signal.

**No inference-economy claim.** This paper proposes no early-exit, routing or
compute-saving mechanism, and none of its AUROCs are strong enough to gate
anything at usable precision. We tested this on the strongest
passively-significant signal the project produced (the SAE feature from
Appendix D's positive control, p=4.8x10^-11): thresholded as a single-feature
classifier it reaches AUROC =0.5614, but its raw activation is at or near zero
for nearly every sample, so no threshold above zero clears even 50% recall at
any of the four target-recall operating points tested, and precision is pinned
at the 4.8% base rate throughout. Extreme statistical significance under
simultaneous testing does not translate into usable gating concentration.

**Data, code and reproducibility.** All code, cached result JSONs and the
paper source are in the anonymized supplementary material, and will be
released as a public repository upon publication. Every result that consumes
only already-cached `results/*.json` artifacts is self-contained; Appendix H
maps each result to the script and artifact that produce it, and lists the
four scripts that additionally need an unshipped sibling repository to rerun
from scratch.

**Broader impact and ethical considerations.** This paper's subject is the
trustworthiness of evidence in interpretability research, so its main
externality is epistemic rather than deployment-facing. Mechanistic claims
derived from interventions increasingly feed arguments about what models
represent internally, and those arguments are cited in safety contexts; a
direction that was never checked against a held-out validity gate can produce
a confident-looking mechanistic story that is indistinguishable, on the
evidence presented, from patching noise -- at the 3:8 holdout used here, a
pure-noise direction passes plausible gates 6.7%-46.1% of the time (§4.3). The
same risk applies to the outcome side: a flip-rate metric contaminated by
degeneracy (over half our completions, §4.2) or floored by a competence
ceiling (3.3%, §4.1) can register a change that is real but mechanistically
uninformative, and in a higher-stakes setting -- an audit, a deployment
decision, a claim that a model's deception or refusal circuitry has been
localized -- that difference matters more than it does here. We therefore
frame the checklist (§5) as a floor rather than a certificate. It is a set of
necessary conditions we can show this paper's own flagship experiment failed;
passing all ten would not establish that a mechanistic claim is correct, and
§5.1 states where our evidence does not reach. The checks are also cheap
enough that reporting them should not, in our view, be treated as a burden
that trades off against publishing negative results -- our own strongest
contributions here are negative. Finally, this work runs only small open
models on a public benchmark and involves no human subjects, no personal data,
and no released capability; the LLM judge labels are model-generated and, as
§6 discloses, were validated only by manual spot-reading and a surface-feature
control, not by independent annotation.

## 7. Conclusion

We set out to test whether an FFN-vs-Attention hallucination asymmetry
analogous to ReDeEP's appears in closed-book generation on small models. The
answer this paper can support is that the question is not answerable on this
testbed with these instruments, and we can now say precisely why.

GPT-2 answers 3.3% of TruthfulQA validation items correctly under an
independent judge, and adding 283 previously-unjudged items added zero correct
answers, so the flip-to-correct outcome variable is floored regardless of any
mechanism. Over half of the nominally hallucinated completions in each
causal-test pool we constructed (51.9%, 53.1%; 53.6% of all baseline
completions regardless of class) are degenerate repetition loops rather than
confabulations, near-balanced across label classes, so the outcome events that
do occur are confounded with loop-breaking. And the difference-of-means
direction the causal test injects never cleared a held-out validity check --
one which, at the 3:8 holdout the causal kernel used, cannot declare any
*observed* AUROC below 0.875 significant. That holdout was set by a per-class
cap rather than by the data: re-scoring the same directions against the 475
negatives the testbed had all along moves the gate's minimum detectable
observable AUROC to 0.779 and shows the directions to be uninformative rather
than anti-informative. The scarce class is the positives: 27 judge-correct
items exist in the entire pool. A label-permutation null for the 200-resplit
version of the same check agrees, where the naive independent-samples reading
of the same numbers would have said the opposite. We therefore do *not* report
the causal null as surviving scrutiny: the instrument provides no
interpretable evidence either way, because the directions it patches never
passed the validity gate.

The passive side is weaker than an earlier version of this work claimed, in a
different way. The peak-versus-peak FFN-vs-Attention margins (0.003-0.011
AUROC) are three to ten times smaller than the spread a single
cross-validation fold seed produces on the same features, and on both 24-layer
models the argmax "peak layer" is not a well-identified quantity at all. A
same-layer paired comparison is stable, but its sign flips between layers
within the same architecture, so which layer is compared determines the
answer. And a leave-one-category-out re-test, with two permutation controls, a
pair-type decomposition and a group-variable-only ceiling calculation,
indicates that a standard random K-fold AUROC for this probe on TruthfulQA
overstates unseen-topic performance by 0.13-0.17 AUROC, with within-topic
performance at the two layers tested at chance and the mechanism behind that
gap unidentified (§4.8).

What we offer instead is transferable: two validity checks (a competence
ceiling measurement and a degeneracy pre-filter) that are cheap,
model-agnostic, and would have changed how we read our own flagship result had
we run them first; an exact minimum-detectable-effect calculation for held-out
direction validation that replaces "n is small" with a number; and a checklist
assembling these with the controls a component-specificity patching claim
requires. Any researcher running an intervention on an LLM and reporting a
flip-rate inherits these same risks, independent of which component or
behavior is being tested.

## Appendix A: Correction History

This appendix is a transparent audit trail of errors found during iterative
internal and external review, and of the checks run to confirm each fix's
effect on reported numbers. The main text states final, corrected numbers
directly; nothing below is required to verify them.

-  **A fabricated-looking claim, removed.** An earlier version claimed "a
   direct FFN-found-vs-Attention-found comparison gives McNemar p=1.000 in
   every configuration." This was never computed:
   `code/01_ffn_causal_patch.py` and its scaled variants save an
   Attention-found flip rate but never run a McNemar test against it, and that
   condition patches the FFN-derived direction at the Attention site rather
   than a direction derived from Attention's own activations, so no version of
   the script could have supported an FFN-vs-Attention specificity claim. The
   genuine test (§4.4) uses a real Attention-derived direction. A related
   paragraph's degeneracy rate ("19.8-29.6% across conditions") was likewise
   only the FFN-found arm at L8 from an earlier, smaller pass, not the full
   range across all twelve conditions it was presented as describing; §4.4 now
   reports the correct range (10.1%-46.5%).

-  **Judge-parser substring bug, fixed and checked.** Every judge-scoring
   function originally classified a verdict as "correct" if `"CORRECT"`
   appeared and `"HALLUCINAT"` did not -- and `"CORRECT"` is a substring of
   `"INCORRECT"`, so "INCORRECT" would have been silently scored as label 1 in
   seven separate implementations. Raw verdict strings were never persisted,
   so this was unfalsifiable from existing artifacts until a review flagged
   it. Fixed in all seven by checking for `"HALLUCINAT"` and `"INCORRECT"`
   first, we reran the full 534-sample GPT-2 relabel end to end:
   byte-identical to the original (same 27/507 split, same kappa=0.0417). We
   also reran the §4.4 flagship causal test (467 test prompts x 17 scored
   conditions -- §4.4's sixteen patched conditions plus the unpatched
   baseline) with the corrected parser: identical at every one of the 467x17
   cells. Caveat: the pre-fix artifacts were overwritten and raw verdict
   strings never persisted, so this comparison cannot be re-verified from the
   released supplement, which ships only post-fix artifacts. The bug was real
   and, on our own check, changed no reported number.

-  **Dosage-mismatch diagnostic, corrected twice.** An earlier version of
   `code/26_ffn_attn_dosage_diagnostic.py` (a) measured on bare TruthfulQA
   questions rather than the "Q: {question}\ nA:" prompts the causal test
   patches, and (b) normalized alpha against each sublayer's own raw output
   norm -- the wrong denominator, since the patching hook replaces that output
   with (out+alpha.direction) and the block's own forward code then adds the
   combined value into the residual stream. The uncorrected version reported a
   2-3x dosage asymmetry between arms; correcting only the prompt format
   shrinks it to 1.3-1.5x; correcting both gives the 16.0%/12.0% (alpha=20)
   and 32.1%/24.1% (alpha=40) numbers in §4.4. Further, because the corrected
   script uses one shared residual-stream denominator for both arms, the
   resulting arm-to-arm equality is true by construction and is now labeled as
   such in §4.4 rather than presented as an empirical finding of "no dosage
   asymmetry."

-  **Surface-baseline computation, flagged as missing.** An earlier version of
   the label-validity discussion asserted that a trivial length/lexical
   baseline does not explain the judge label's structure at chance-level AUROC
   -- no such computation existed in this project at the time.
   `code/32_surface_baseline_vs_judge_label.py` was written to actually
   compute it; the result (not chance-level, and not meaningfully different
   from the same baseline against the Jaccard label) is reported in §6.

-  **Direction-validity CIs misdescribed, and a 10x cosine transcription
   error.** (a) The four direction-validity bootstrap CIs at n=11 were
   described as "all consistent with chance"; three of the four in fact
   exclude 0.5 entirely, and an exact Mann-Whitney test finds them nominally
   significant in the anti-predictive direction (§4.3); a further correction
   this round made those p-values consistently two-sided (L8 FFN/Attn 0.0485,
   not the one-sided 0.0242 quoted before). (b) The permutation-based
   cosine-similarity check reported "-0.058/-0.051"; the result file gives
   -0.0058/-0.0051, which is the tier-1 kernel's own pair of directions (fit
   on 47 items) and a different quantity from the -0.054/-0.056 in §4.4 and
   §5, the cosine between the FFN and the genuinely Attention-derived
   direction in the component-specificity kernel (fit on 58 items). Both are
   correct for their own run. Fixing (a) exposed that the power analysis used
   to argue the test was uninformative (`code/43`) was itself one-sided,
   structurally unable to detect power in the direction the data fall in.
   Recomputed two-sided, the test is reasonably well-powered (73-99%) at the
   observed AUROCs, which motivated the 200-resplit diagnostic
   (`code/46_direction_validity_resplit_diagnostic.py`) in §4.3; that
   diagnostic showed the original seed's anti-predictive result to be an
   atypical low-tail draw, not a stable property of these directions.

-  **Random-direction-ensemble percentiles, misread as informative.** An
   earlier version described the random-direction ensemble as "the strongest
   single check" and reported the found directions at the "50th" and "40th"
   percentile of a 20-draw ensemble. Both the framing and the reading were
   wrong: the found directions produced 0/60 flips, and 10 of 20 random FFN
   directions and 8 of 20 random Attention directions also produced 0/60, so
   those percentiles are tie counts at the statistic's floor. The same applies
   to the enlarged-pool rerun's "95th"/"100th" percentiles, which are driven
   by 2/60 and 3/60 flip events. §4.4 now reports raw counts and states that
   this check is uninformative at this flip-rate floor.

-  **A CV-seed effect misattributed to an aggregation convention.** An earlier
   version explained the gap between two AUROCs reported for the same GPT-2 L8
   FFN features (0.6053 and 0.643) as a difference between mean-of-folds and
   pooled out-of-fold aggregation, and separately explained a peak-layer
   disagreement for Qwen0.5B as the two aggregation conventions "genuinely
   disagreeing." Both attributions were wrong. `code/02` uses
   `StratifiedKFold(random_state=42)` and `code/37` uses `random_state=0`; the
   seed accounts for +0.0365 of the +0.0374 total gap and the aggregation
   convention for +0.0010 (`code/50_cv_seed_sensitivity_sweep.py`). §4.5 now
   reports the decomposition and the resulting fold-seed sensitivity band, and
   attributes the peak-layer instability to seed sensitivity.

-  **A category-leakage claim, twice revised and finally narrowed.** An
   earlier version reported, in its abstract and headline results, that a
   leave-one-category-out CV re-test collapsed the component probe's AUROC
   from 0.62-0.66 to 0.48-0.49, attributed this to category-clustering leakage
   (specifically, the probe learning per-topic correct-answer rates), and
   extended the claim to a "heterogeneous" across-architecture leakage result.
   An independent review then argued the entire effect was an artifact of the
   LOGO estimator's per-category averaging at n_+ 1.7. We ran the controls
   that separate these (`code/48_permuted_pseudocategory_control.py`); *both*
   readings were wrong, in different directions. (a) Two permutation controls
   (size-matched, and size- and class-matched) show random groupings recover
   0.58-0.62, not chance, so the estimator is not biased downward at this n
   and the collapse is specific to real topic structure (0-10 of 1,000 matched
   draws fall at or below the real value across the four cells; all four
   survive Holm correction). We predicted the artifact outcome before running
   this control and were wrong. (b) A group-variable-only ceiling calculation
   shows the originally claimed mechanism cannot be demonstrated: topic
   identity alone yields AUROC 0.7938 in-sample but 0.5054 leave-one-out and
   0.4863/0.4789 under this paper's own 5-fold CV. We initially wrote this up
   as ruling the mechanism out; that too was an overreach, since the in-sample
   value exceeds the probe's own AUROC and the out-of-fold shortfall is an
   estimation-noise problem at 27 positives over 38 categories. The text now
   says only that the account cannot be tested here. (c) A third check, added
   last, decomposes the standard CV's own out-of-fold scores by pair type and
   shows that LOGO silently changes the estimand: only 4.9% of standard-CV
   pairs are within-topic. (d) **That third check was then reported wrongly,
   in a way an external reviewer caught, and its correction is the largest
   single change in this revision.** We wrote that "the remaining drop -- from
   0.54-0.68 with same-topic items in training to 0.48-0.49 without them -- is
   the part attributable to training-set topic overlap," and called it
   "substantial (0.06 to 0.20 AUROC)." That subtraction was invalid: 0.54-0.68
   is a *pair-weighted* pooled AUROC over 675 within-topic pairs
   (`code/48::within_between_topic_auroc`), while 0.48-0.49 is an *unweighted
   mean of 16 per-category* AUROCs (`code/47::probe_leave_one_category_out`);
   differencing them mixes the effect of removing same-topic training data
   with the mechanical effect of changing the weighting. Recomputing the
   within-topic column under LOGO's own averaging convention, from the same
   standard-CV out-of-fold scores
   (`code/52_estimand_matched_within_topic_auroc.py`), gives 0.4674, 0.5134,
   0.4408 and 0.5497, so the estimand-matched training-overlap residual is
   -0.011, +0.023, -0.041, +0.061 -- negative in two of four cells and not
   significant in any (paired p=0.15-0.85). The claimed effect was entirely
   the estimand mismatch. The corrected statement is stronger than the one it
   replaces: this probe's within-topic discrimination is at chance whether or
   not same-topic items are in training, so no training-overlap effect could
   have been observed. The abstract, §4.8 and checklist item 10 are rewritten
   accordingly. (e) The permutation controls in (a) were run at 100 draws,
   whose attainable two-sided p floor (0.0198) is coarser than the smallest
   Holm threshold across the four cells (0.0125), so no cell could have
   survived correction regardless of the data. They are rerun at 1,000 draws
   and Holm-corrected across the four cells; §4.8 reports what survives. The
   surviving claim is that standard random K-fold CV overstates unseen-topic
   performance for this probe by 0.13-0.17 AUROC, with the mechanism
   unidentified. The cross-architecture extension is removed regardless, being
   confounded by three simultaneous protocol differences aliased with
   architecture.

-  **"The n=11 holdout this testbed supports" was a design choice described as
   a data property.** The abstract, §4.3, §5 item 3 and the Conclusion all
   characterized the direction-validity gate's 3:8 holdout as what the testbed
   could supply. It is what `TRAIN_N_PER_CLASS`,=40 in
   `kaggle_kernels/paper1-causal-patch-tier1-validated/` supplies: the
   training pool is capped at 40 items per class and the holdout is the last
   20% of each class, giving 8 negatives, while the judge-labeled pool holds
   507 hallucinated items of which only 40 are used at all and only 32 spent
   on fitting the direction. Re-scoring the identical directions against all
   475 eligible negatives (`code/54_enlarged_negative_holdout_gate.py`,
   verified to reproduce the kernel's 3:8 values first) improves every
   reported gate characteristic at zero data cost and changes a substantive
   reading: the four held-out AUROCs move from 0.083/0.083/0.0/0.125 (three
   nominally significant in the anti-predictive direction) to
   0.393/0.304/0.277/0.183 with all four exact two-sided p>=0.056 --
   uninformative, not anti-predictive. §4.3 now reports the exact power
   calculation along both class axes (Table 3) and states that positives (27
   in the entire pool) are the binding constraint.

-  **A conclusion asserted without its test: the 200-resplit diagnostic.**
   §4.3 previously described the resplit means as "0.54-0.58 centered at or
   slightly above chance" and drew a conclusion without testing it. Because
   the 200 resplits reuse the same 27 positives, a reader computing a naive
   independent-samples z from the reported mean and SD would obtain
   z=2.65-5.62 and conclude the opposite of what the paper claims. We now run
   the correct test -- a label-permutation null rerunning the entire
   200-resplit procedure under 2,000 permutations of the judge labels
   (`code/53_resplit_permutation_null.py`) -- and report it in §4.3 (Table 1).
   It supports the paper's conclusion (no cell reaches p<0.05; p=0.085-0.215)
   and shows the correct standard error to be 3.4-3.6x the naive one. The
   conclusion did not change; it went from asserted to demonstrated.

-  **Small numeric and labeling corrections made in this round.** (a) The
   GPT-2 534-item judge-vs-Jaccard kappa was stated as 0.03 in one place; the
   correct value is 0.0417 (0.03 is Pythia's kappa, mislabeled) (§3, §6). (b)
   Pythia's Jaccard-label peak FFN AUROC was stated as 0.615 in one place and
   0.6181 in another; the artifact-verified value is 0.6181 (§4.6). (c) Two
   different minimum-detectable-odds-ratio ranges (3.75-8.0 and 3.25-6.0, both
   at n=467) were presented as though they described the same test; they are
   the native-site FFN-vs-Attention comparison (directions fit on 58 items)
   and the common-site comparison (directions fit on 47 items) respectively,
   and are now labeled (§4.1, §4.4). (d) The TOST equivalence-bound range
   (§4.4) was stated as 2.85-10.9; the maximum achieved in the underlying grid
   search is 10.85 (`results/tier1_tost_competing_risks.json`). An earlier
   version of this correction added that "10.9 is not a grid point"; that
   justification was itself wrong -- `code/38` searches `np.arange(1.05, 15.0,
   0.05)`, which does contain 10.90 -- the number is simply not the one the
   search returns. (e) The low-dose sweep range (§4.4) was stated as 42-52%;
   the computed range is 41.67-51.67%, and "no monotone dose-response" was
   inaccurate: the Attention arm is strictly monotone decreasing across the
   three doses, so the accurate statement is "no monotone increase." (f) The
   enlarged-pool replication was described as "nearly double the power" at
   n=750 versus n=467; the ratio is 1.61x, and (per §4.1) discordant-pair
   counts, not n, determine resolution.

-  **Further corrections made in this revision.** (a) **A leakage instance in
   our own code.** `code/37_paired_component_delta_auroc.py` fit its
   `StandardScaler` on the full dataset before `cross_val_predict` rather than
   inside a per-fold `Pipeline` as `code/02`, `code/47`, `code/49` and
   `code/50` all do -- a real leak in the script producing §4.5's paired-Delta
   and BCa results. Now fixed and `code/37` rerun: the effect is -5.6x10^-5
   AUROC at the GPT-2 L8 FFN cell, every Delta and BCa interval is unchanged
   at the precision printed, no interval changes whether it covers zero, and
   two naive-peak values move in the third decimal (0.632to0.633 GPT-2 Attn,
   0.563to0.562 Qwen0.5B FFN). Separately, §4.5 had described the
   `code/02`-vs-`code/37` gap as decomposing into exactly two terms (fold seed
   and aggregation convention); there are three, the third being this scaler
   placement, and §4.5 now states all three. (b) **An undisclosed selection
   rule.** §4.1 reported "0 of 283" newly-judged items correct without noting
   that those 283 were exactly the items whose completions scored <=0.12
   Jaccard overlap against *every* correct and *every* incorrect reference
   answer, which is why the older filter could not label them; that
   pre-selection makes 0/283 much less surprising than it reads unqualified.
   §4.1, the abstract and the Introduction now state the rule. (c) **"Only
   prompt construction changed" was false.** §4.6's Qwen0.5B bare-versus-chat
   comparison spans different samples (N=513 vs N=433, only 298 questions
   shared), because the labeling filter depends on the completion; about half
   the base-rate difference is sample composition (49.3% vs 57.5% overall;
   53.7% vs 58.1% on the shared 298). §4.6 now discloses this and downgrades
   the template-reversal claim. (d) **A leftover fragment of the abandoned
   mechanistic thesis.** §4.5 contained a sentence reading a single L8
   direct-logit-attribution difference (5.08 vs 4.85, in-sample, no SDs, 1 of
   24 cells, with a larger opposite-signed same-layer Attention effect) as
   "suggestive of an over-retrieval signature" -- precisely the cherry-picked
   single-cell inference this paper argues against. The sentence is removed.
   (e) **Related work asserted without citations, and five references without
   authors.** The paragraph naming this paper's closest relatives ("protocol
   audits") cited nothing; it now engages Hewitt and Liang (2019), Janiak et
   al. (2025) and Hussain and Kantarcioglu (2026) specifically. Five
   bibliography entries previously appeared by title only; all now carry
   verified author lists. One entry asserted a venue we could not verify
   ("ParamMute, NeurIPS 2025"); it is cited as a preprint. (f) **An
   uncontrolled negative R^2.** Appendix E read the entropy head's negative
   held-out R^2 under gradient reversal as evidence that "the adversarial
   pressure works," with no non-adversarial baseline to distinguish that from
   the head never having learned to predict entropy at all. `code/17` now
   trains the identical head at lambda=0 (no pressure, R^2=+0.199/+0.115) and
   lambda=-1 (cooperative, R^2=+0.542/+0.362), and Appendix E reports all
   three. The original reading survives the control; it had simply not been
   earned. (g) **Checklist item count stated two ways.** The abstract said
   "eight-item" and the Contributions said "ten-item eight causal-patching,
   two passive-probing." Both were literally true of different subsets; the
   abstract now uses the ten-item phrasing. (h) **A de-anonymization vector in
   the supplement.** The anonymized supplementary archive contained a
   plain-text file with this project's git commit hash, which would identify
   the repository the moment it became public. It is removed from the archive.

-  **Correction confessions relocated.** Earlier drafts narrated several of
   the above corrections inline in the results sections. All are now collected
   here and the main text states final numbers directly; where a reader might
   otherwise reconstruct the wrong quantity, §4.3-§4.6 and §5 carry a
   one-clause pointer to the relevant item above rather than a retelling. The
   single deliberate exception is §4.8, whose subject *is* how its own
   interpretation changed under two permutation controls, a pair-type
   decomposition and a ceiling calculation; that history is in the body
   because the sequence of checks, not the final number alone, is the
   transferable result.

## Appendix B: Causal-Patching: Resolution, Equivalence and Supporting Controls

**What this null can and cannot exclude.** Discordant pairs per configuration
are 19, 15, 9, 16. At those counts, the minimum odds ratio an exact two-sided
McNemar test can detect at p<0.05 is 3.75 (L8/alpha=20), 4.00 (L8/alpha=40),
8.00 (L9/alpha=20), 4.33 (L9/alpha=40). **This 3.75-8.00 range belongs to the
native-site FFN-found-vs-Attention-found test, whose directions are fit on the
full 58-item train pool**, and is a different quantity from the 3.25-6.00
range in §4.1, which is the common-injection-site comparison from the tier-1
validated kernel, whose directions are fit on the 47-item direction-fit subset
remaining after the validity holdout. Both are at n=467 prompts; they differ
in comparison, injection site and direction-fit set, and their discordant
counts differ accordingly (19/15/9/16 versus 17/17/14/15). The observed odds
ratios and approximate 95% CIs (log-odds normal) are 0.46 [0.18, 1.21], 0.67
[0.24, 1.87], 1.25 [0.34, 4.66], and 3.00 [0.97, 9.30]. Only L9/alpha=20
([0.34,4.66]) is simultaneously consistent with FFN being several-fold worse
and several-fold better than Attention; the other three exclude one direction
or the other at the several-fold scale while still failing to exclude a
moderate effect. These are properties of the observed discordant counts, not
evidence about the two components -- see "the honest reading" below for why no
reading of these intervals is licensed.

**Equivalence testing.** A TOST search finds no cell establishes equivalence
at the pre-registered OR=2.0 bound; the smallest achievable equivalence bounds
across the eight tested cells (native and common site x four configurations)
range 2.85 to 10.85 (`code/38_tier1_tost_competing_risks.py`). Both sites here
are the *tier-1* kernel's, whose directions are fit on the 47-item subset and
whose native-site discordant counts are 17/13/17/18 -- again not the
19/15/9/16 test above. This test cannot assert equivalence either.

**Injection-site and dosage controls.** A common-injection-site test (patching
both directions at the shared post-block residual stream) confirms
FFN-native-site and FFN-common-site patching are mathematically identical for
GPT-2's block structure (verified byte-for-byte), and finds the genuinely new
Attention-common-site comparison non-significant at every configuration
(p=0.1435, 0.049, 0.7905, 0.6072; the nominal 0.049 does not survive Holm
correction across even these four tests). A 2x3 competing-risks (flip /
no-flip / degenerate) chi-square finds one nominal p=0.0097 that likewise does
not survive correction. On dosage, measured on 100 correctly-formatted
prompts, the residual-stream norm at the injection point is 124.7 (L8) and
166.0 (L9), giving relative perturbations of 16.0%/12.0% at alpha=20 and
32.1%/24.1% at alpha=40 (`code/26_ffn_attn_dosage_diagnostic.py`). **This
quantity is equal for the FFN-site and Attention-site arms by construction,
not by measurement**: the script uses a single shared denominator (the
residual-stream norm at that layer) for both arms, so their relative
perturbation is necessarily identical. What the diagnostic establishes is that
the *correct* denominator is the residual stream rather than each sublayer's
own output norm (Appendix A item 3). It does not independently demonstrate the
absence of a dosage asymmetry, because with this denominator no asymmetry
could have been observed.

**A low-dose sweep.** At the common site with alpha in 2.5,5,10 (Jaccard
label, for speed), flip rates are flat: FFN 50.00%, 45.00%, 51.67%; Attention
50.00%, 48.33%, 41.67% -- a 41.67-51.67% range overall. There is **no monotone
increase** with dose in either arm; the Attention arm is in fact monotone
*decreasing* across the three doses, the opposite of what a genuine
dose-responsive steering effect would produce.

**A differential-degeneration confound in the outcome metric itself.** A
non-trivial fraction of interventions in every condition produce a degenerate
or unparseable completion rather than a clean correct-or-wrong answer, ranging
10.1%-46.5% across all twelve found/random/attn-found conditions at both
layers and alphas, in the n=228 Jaccard-label scaled run
(`results/ffn_causal_patch_scaled_results.json`) rather than the n=467
validated-label run this subsection otherwise reports. Found and random
directions sit at broadly similar rates in most configurations but diverge at
L8, where the found direction degenerates *more* than a random direction of
equal norm (alpha=20: 25.9% vs. 16.7%; alpha=40: 40.8% vs. 25.4%). This does
not generalize: at L9 the ordering reverses (alpha=20: 11.0% vs. 15.4%;
alpha=40: 19.3% vs. 22.8%). We note the L8 pattern because it is consistent
with the flip-rate signal being generic perturbation rather than targeted
correction, but two of four cells run the other way. Because unparseable
completions are scored as not-flipped in every condition, this differential
degeneration could mechanically penalize whichever arm degenerates more; we do
not attempt a competing-risks correction and flag it as unresolved.

**Extension beyond GPT-2.** The same test on Pythia-410M and
Qwen2.5-0.5B-Instruct (chat-templated) is too underpowered to support any
claim: on Pythia valid pairs collapse from n=22 at alpha=10 (p=0.25) to n=7 at
alpha=20 to n=0 at alpha=40 (generation degenerates entirely), and on
Qwen0.5B-chat valid pairs are n=2,1,0 at alpha=10,20,40 -- uninformative
rather than merely underpowered, since this instruction-tuned model's
chat-style responses rarely clear the word-overlap labeling threshold even at
baseline.

## Appendix C: Cross-Architecture Passive Results: Sample Composition and Relabeling

**More than the prompt changed between these two conditions.** They are also
different samples: N=513 (bare) versus N=433 (chat), with only 298 questions
common to both (215 bare-only, 135 chat-only), because the Jaccard filter that
admits an item to each pool depends on the completion, which the template
changes. Sample composition accounts for roughly half of the apparent
base-rate shift: the Jaccard-correct rate is 49.3% (bare, n=513) versus 57.5%
(chat, n=433), an 8.2 point gap, but on the 298 shared questions it is 53.7%
versus 58.1%, a 4.4 point gap. We did not recompute the component probe on the
matched 298 -- the chat condition's per-sample activations were not cached --
but a composition difference of this size is of the same order as the 0.0284
peak margin the reversal turns on. **The template-reversal result should
therefore be read as suggestive and confounded with sample composition, not as
a clean single-variable manipulation** (Appendix A item 12c). The finding
below that the same reversal does not survive relabeling under the judge label
is the stronger reason not to lean on it.

**Re-probing under the validated label.** We relabeled every completion on all
three architectures with the LLM judge and reran the component probe under
both labels (`code/24_llm_judge_score_all_architectures.py`,
`results/llm_judge_relabel_summary.json`), and separately extended the check
to GPT-2 itself (`code/29_gpt2_full_validated_relabel_rerun.py`, all 534
samples). Absolute AUROCs rise substantially on every architecture (GPT-2
0.605/0.616to0.698/0.717 FFN/Attn peak; Pythia 0.6181/0.6115to0.7353/0.7494;
Qwen0.5B-bare 0.5657/0.5625to0.7127/0.6992; Qwen0.5B-chat
0.5704/0.5988to0.6603/0.6412), and FFN's numerical majority is restored on
Qwen0.5B-chat (11/24 under Jaccard to 18/24 under the validated label), so the
template-reversal finding is itself label-sensitive. Under the validated label
FFN's majority *rises* on three of four architecture/template conditions
(Pythia 16/24to18/24; Qwen0.5B-bare 14/24to20/24; Qwen0.5B-chat 11/24to18/24)
and only GPT-2 moves the other way, reversing further toward Attention
(8/12to2/12). FFN holds a layer-count majority on 3 of 4 conditions under
either label, but with different members (GPT-2 flips out as Qwen0.5B-chat
flips in). Which component *peaks* is less stable still: only 2 of 4
conditions keep the same peak winner under both labels (GPT-2: Attention both
times; Qwen0.5B-bare: FFN both times), and the other two flip in opposite
directions.

## Appendix D: Two Further Causal Instruments

**ROME-style causal tracing.** Additive mean-shift steering is a comparatively
weak causal instrument. We replace it with causal tracing (Meng et al. 2022),
adapted to closed-book QA by corrupting the whole question span, since no
single clean "subject span" applies: a clean run scores a forced-choice
logit_diff between the correct and incorrect reference answer's first token; a
corrupted run adds Gaussian noise to the question-span embeddings; a
restoration sweep patches each (layer, component) one at a time; a specificity
control repeats this with a mismatched example's activation
(`code/08_rome_style_causal_tracing.py`).

At the maximum powered sample (n_valid=67, pre-registering the joint 24-test
correction as primary), FFN shows no specific restoration effect anywhere.
Attention's smallest-p candidate (L9, own-shuffled =+0.151; attn L7's effect
is larger at +0.163 but its p=0.0146 is not) does not survive the joint
Holm-Bonferroni threshold (p=0.012 vs. 0.05/24=0.00208) -- attenuated from an
earlier, lower-powered n=45 pass where the same cell was larger (+0.214,
p=0.0026) under a less conservative per-family scoping. The FFN site at L9
(`mlp_9` in the tracing artifact, the same sublayer called FFN throughout)
instead gives an *anti-specific* result that clears the strict joint threshold
(p=0.00086, own-shuffled =-0.203): a mismatched example's activation restores
discrimination *better* than the example's own. The surviving cell runs the
wrong way, so this is not in tension with "no specific restoration effect
anywhere," and we do not claim it is a stable finding rather than one more
stopping point in a noisy series. Neither Pythia-410M nor Qwen2.5-0.5B shows
any layer or component surviving correction under either framing (Pythia's
smallest uncorrected p=0.0645, Attn L20; Qwen's p=0.0041, Attn L15).

Rerun under the validated judge label
(`code/40_rome_style_causal_tracing_validated.py`), averaging 10 independent
corruption draws per example and ensembling the mismatched-donor control over
10 random donors, the judge-correct pool is small (17 usable candidates out of
27 total judge-correct GPT-2 samples, after requiring a parseable
correct/incorrect answer pair) -- the same competence ceiling as §4.1,
reappearing in a different instrument. Across individual corruption draws,
only 53.5% show real degradation from corruption, underscoring why averaging
over draws rather than selecting on one is the right correction. No cell
survives Holm-Bonferroni at either this test's own 24-comparison family or a
stricter 120-test family across all three architectures; the two smallest
uncorrected p-values are both Attention layers (L10 p=0.026, L11 p=0.023). One
limitation is disclosed rather than fixed: the forced-choice score remains
first-token log-odds, not a length-normalized full-sequence log-probability.

**A sparse-feature intervention.** A dense mean-difference direction averages
over every latent factor that differs between conditions; if a correction
signal exists but is carried by a small number of sparse features, additive
steering along the dense direction would dilute it. We substitute
`jbloom/GPT2-Small-SAEs-Reformatted`'s layer-8 SAE (d_sae=24,576, trained on
300M tokens of OpenWebText) -- a disclosed mismatch on two axes at once: wrong
hookpoint (residual stream, not FFN-sublayer output) and wrong training
distribution. **0 of 24,576 features survive Benjamini-Hochberg FDR at q=0.05
on this paper's own 534-example dataset**
(`results/sae_feature_clamp_paper1.json`), so the causal clamp step was never
reached. This is a null one stage earlier than §4.4's test, but bounded by
instrument mismatch: either axis alone could explain zero surviving features.
Running the identical procedure on a companion dataset (HaluEval, n=500, same
SAE, same layer) as a positive control
(`code/15_sae_feature_gating_utility.py`) finds 331/24,576 features surviving
FDR (best p=4.8x10^-11), confirming the null above is not a feature-selection
bug. Even there, the causal clamp shows no specificity at any strength
(p=1.000, 0.508, 1.000 at eta=10,20,40;
`results/sae_feature_clamp_combined.json`).

## Appendix E: Difficulty-Matched and Adversarial Controls

The 0.03 AUROC margin over a surface-feature baseline (0.605 vs. 0.576) is
small enough that a generalized question-difficulty signal remains a live
alternative to a hallucination-specific one. We test this two ways on GPT-2,
matching correct/hallucinated groups within 10 quantile bins of (1) mean
token-level output entropy and (2) the out-of-fold predicted probability of a
logistic regression over all 6 surface features, then re-probing the identical
FFN/Attn comparison on each matched set
(`code/06_difficulty_matched_control.py`). Entropy-matching retains n=492/534;
the composite match retains n=462/534. Neither proxy correlates significantly
with correctness before matching (entropy r=0.045, p=0.295; composite r=0.024,
p=0.578) -- this dataset never had a statistically detectable difficulty
confound for either proxy to remove, so this is survival under a weak test,
not dissociation under a strong one.

The signal survives both matches essentially unchanged (entropy-match: FFN
0.6085+/-0.0442, Attn 0.6102+/-0.0285; composite-match: FFN 0.6255+/-0.0510,
Attn 0.6253+/-0.0926; vs. 0.6053+/-0.0557 / 0.6165+/-0.0427 unmatched), with a
label-permutation test (500 shuffles) giving both components p=0.0020 -- the
permutation floor at this shuffle count. Extending to the other architectures
(`code/11_multi_arch_difficulty_matched_control.py`) splits by architecture:
Pythia replicates GPT-2's survival (FFN 0.6327+/-0.0513, Attn 0.6093+/-0.0221,
both p=0.0020, 89.3% retention), Qwen0.5B does not (FFN 0.5277+/-0.0434,
p=0.230; Attn 0.5282+/-0.0320, p=0.206, 88.3% retention).

A gradient-reversal probe attaches a second head predicting the entropy proxy
from the same shared representation, back-propagating its *negative* gradient
(`code/17_gradient_reversal_adversarial_probe.py`). Held-out R^2 for the
entropy head is negative for both components (-0.71 FFN, -1.44 Attn). **A
negative R^2 on its own does not show that the adversarial pressure worked**
-- it is equally consistent with a head that never learned to predict entropy
at all -- so we run the same head without the pressure. At lambda=0 (the
entropy head learns but sends no gradient to the encoder) R^2 = +0.199 (FFN)
and +0.115 (Attn); trained cooperatively at lambda=-1 (the encoder optimized
to *help* predict entropy, an attainable ceiling) it reaches +0.542 and
+0.362. The head can therefore predict this entropy proxy from these
representations, and the adversarial condition drives it far below its own
unopposed baseline -- which is what licenses reading the negative values as
suppression. Hallucination AUROC nonetheless survives above chance for both
(FFN 0.5944+/-0.0624, Attn 0.6134+/-0.0542, both p=0.0050 against a
200-shuffle floor).

Rerunning the matched controls under the validated label
(`code/30_difficulty_matched_control_judge_label.py`) is inconclusive: the
severe class imbalance leaves only 54/534 samples (10.1%) after matching, 27
per class, roughly 5 per CV fold, and the two proxies then disagree
(entropy-only: FFN 0.7840, p=0.0060; Attn 0.7187, p=0.0200; composite: FFN
0.3987, p=0.82; Attn 0.3680, p=0.89). We read neither as informative. This
control also probes the Jaccard-label peak layers (FFN L8, Attn L3) rather
than the validated label's own shifted peaks -- a layer mismatch we did not
correct for. The difficulty-dissociation result should therefore be read as
established only under the Jaccard label.

## Appendix F: Layer Localization, and its Label-Dependence

For context on where the probes above are read, we report GPT-2 layer
localization from the vendored mech-int artifacts (verified byte-identical to
the originating project's logs; `code/00_verify_vendored_mechint_numbers.py`
reprints each). Under the Jaccard label, six methods converge on layers 8-9:
dense probe peak L9 (0.5827); sparse L1 probe peak L9 (100/768 active dims,
87% sparse, CV AUROC 0.589, not the inflated in-sample 0.874); token-position
probe peak L8 last-token (0.6036); component probe FFN peak L8 (0.6053) vs.
Attn peak L3 (0.6165); steering peak improvement at L9; gold-token logit-lens
divergence at L8. A seventh method, DLA magnitude, does not join: L10 (+0.90)
and L11 (+0.71) are larger than L9.

Three of the six are weaker than "converges" implies. The dense probe's L9
peak is a near-tie: 0.5827275 at L9 against 0.5826826 at L12, a margin of
4.5x10^-5 AUROC -- about a thousandth of that layer's own cross-validation
standard deviation (0.0497) -- with L10 (0.5805) and L11 (0.5791) also within
0.004. Steering's "peak improvement" at L9 is 0.0015 AUROC points, the argmax
over a 52-cell grid (13 layers x 4 steering strengths) in which the
improvement is exactly zero at 11 of the 13 layers, the only other non-zero
cell being +0.0001 at L10; against a between-layer spread of 0.520-0.609 in
the unsteered baseline AUROC, an argmax over a statistic that is identically
zero almost everywhere localizes nothing. (The sweep artifact records no
per-layer cross-validation standard deviations, so we report no noise band for
it.) And the logit-lens analysis computes two divergence-layer estimates from
the same run: the plain correct-vs-hallucinated divergence peaks at layer 1,
and only the gold-token variant peaks at L8.

**This convergence does not survive relabeling.** Rerunning the four methods
that consume only cached activations under the full-534 judge label
(`code/29_gpt2_full_validated_relabel_rerun.py`, same probing code, only the
label array swapped), the peak layer moves: dense probe L7 (0.6444), sparse L1
probe L7 (now 19/768 active dims, 98% sparse, CV AUROC 0.6260), token-position
probe L6 (0.6961), DLA's largest FFN difference from L10 (+0.90) to L11
(+1.81). L8-9 is a property of the Jaccard label's noise pattern on this
dataset, not a label-independent localization. Read together with the
class-imbalance caveat in §4.6 and the fold-seed sensitivity in §4.5, the
specific peak layers under either label should be treated as unstable, not as
competing precise estimates.

## Appendix G: Leave-One-Category-Out: Estimand Decomposition and Scope

- Cell | pooled OOF | between-topic | within-topic | within-topic | LOGO |
  matched

- (13,689 pr.) | (13,014 pr.) | pair-wt. (675 pr.) | cat.-avg. (16 cat.) | (16
  cat.) | residual (p)

- L8 FFN | 0.6152 | 0.6193 | 0.5363 | 0.4674 | 0.4788 | -0.011 (0.85)

- L8 Attn | 0.6153 | 0.6163 | 0.5970 | 0.5134 | 0.4907 | +0.023 (0.75)

- L9 FFN | 0.6067 | 0.6094 | 0.5541 | 0.4408 | 0.4816 | -0.041 (0.62)

- L9 Attn | 0.6494 | 0.6475 | 0.6844 | 0.5497 | 0.4891 | +0.061 (0.15)

*Where the standard-CV AUROC's discrimination lives, and the estimand-matched
training-overlap residual. Columns 2--5 are all computed on the identical
out-of-fold predicted probabilities from standard 5-fold CV. "within-topic
(pair-wt.)" pools all 675 within-topic (positive, negative) pairs;
"within-topic (cat.-avg.)" instead averages per-category AUROCs over the same
16 categories LOGO uses, which is LOGO's own averaging convention. Only the
latter is estimand-matched to the LOGO column, so only those two form a valid
subtraction. The residual is (cat.-avg.) - LOGO, i.e. what adding same-topic
items back into training is worth; p is a paired two-sided t-test across the
16 categories (exact Wilcoxon signed-rank gives 0.16-0.70, the same
conclusion).*

**Scope of this check, and what we removed.** It covers GPT-2 only, at L8/L9,
for the component probe, under the judge label, with 16 usable folds; we did
not run it for the other layer-localization methods in Appendix F. An earlier
draft extended the same LOGO check to Pythia and Qwen0.5B and reported a
"heterogeneous," architecture-dependent leakage result; **we removed that
analysis rather than repair it**, because it was confounded independently of
anything above -- the GPT-2 diagnostic used last-token extraction, judge
labels and a 5.1% positive rate, while the cross-architecture kernel used
mean-pooled extraction, Jaccard labels and a 47% positive rate, three protocol
differences fully aliased with architecture. The files are retained for the
audit trail and are not cited as evidence.

## Appendix H: Reproducibility Map

Four scripts require an unshipped sibling repository to *rerun from scratch*
(as opposed to re-verifying saved outputs): `code/01_ffn_causal_patch.py`
imports live code from a `mech-int` sibling project to regenerate labeled
completions; `code/03_fisher_geometry_ffn_attn.py`,
`code/06_difficulty_matched_control.py` and the feature-cache step of
`code/49_nondegenerate_subset_probe.py` depend on that project's 2.9GB
`activations.pkl`; `code/15_sae_feature_gating_utility.py` reads cached hidden
states from a second sibling project. Cached intermediate feature files
(`*.npz`) are excluded from the released repository and regenerated by the
scripts that consume them; consequently
`code/50_cv_seed_sensitivity_sweep.py`, which reads GPT-2 features from a
cache written by `code/49`, inherits the unshipped `activations.pkl`
dependency for its GPT-2 rows in a from-scratch rerun, while its Pythia and
Qwen rows do not.

Entries below are ordered by this paper's emphasis: the validity checks first,
the mechanism experiments second.

itemize 0pt0pt

-  **Competence ceiling (27/817; 0/283 newly judged)** (§4.1):
   `kaggle_kernels/paper1-causal-patch-enlarged-pool/` ->
   `kaggle_kernels/paper1-causal-patch-enlarged-pool/output/causal_patch_enlarged_pool_results.json`

-  **Degeneracy check and non-degenerate re-probe** (§4.2):
   `code/04_degeneration_check.py`,
   `code/14_causal_patch_scaled_degeneration_filter.py`,
   `code/49_nondegenerate_subset_probe.py` ->
   `results/ffn_causal_patch_scaled_degeneration_filtered.json`,
   `results/nondegenerate_subset_probe.json`

-  **Direction-validity gate: exact MDE/power table** (§4.3, §5):
   `code/51_direction_validity_mde_table.py` ->
   `results/direction_validity_mde_table.json`

-  **Direction-validity gate at the available negative count** (§4.3):
   `code/54_enlarged_negative_holdout_gate.py` ->
   `results/enlarged_negative_holdout_gate.json`

-  **Direction-validity gate, 200-resplit diagnostic** (§4.3):
   `code/46_direction_validity_resplit_diagnostic.py` ->
   `results/direction_validity_resplit_diagnostic.json`

-  **Label-permutation null for the 200-resplit diagnostic** (§4.3):
   `code/53_resplit_permutation_null.py` ->
   `results/resplit_permutation_null.json`

-  **Direction-validity gate, alternative estimators; one-/two-sided power**
   (§4.3): `code/44_alternative_direction_estimators.py`,
   `code/43_direction_validity_power_analysis.py` ->
   `results/alternative_direction_estimators.json`,
   `results/direction_validity_power_analysis.json`

-  **Permuted-pseudo-category control and topic-only AUROC ceiling** (§4.8):
   `code/48_permuted_pseudocategory_control.py` ->
   `results/permuted_pseudocategory_control.json`

-  **Estimand-matched within-topic AUROC and training-overlap residual**
   (§4.8): `code/52_estimand_matched_within_topic_auroc.py` ->
   `results/estimand_matched_within_topic_auroc.json`

-  **Original LOGO-CV diagnostic (interpretation revised by the controls
   above)** (§4.8): `code/47_category_leakage_diagnostic.py` ->
   `results/category_leakage_diagnostic.json`

-  **CV fold-seed sensitivity sweep and gap decomposition** (§4.5):
   `code/50_cv_seed_sensitivity_sweep.py` ->
   `results/cv_seed_sensitivity_sweep.json`

-  **Causal patching, base runs** (§4.4): `code/01_ffn_causal_patch.py`,
   `code/10_ffn_causal_patch_scaled.py` ->
   `results/ffn_causal_patch_results.json`,
   `results/ffn_causal_patch_scaled_results.json`

-  **Causal patching under the validated label** (§4.4):
   `kaggle_kernels/paper1-causal-patch-judge-label/` ->
   `results/causal_patch_judge_label_results.json`

-  **Causal patching beyond GPT-2 (Pythia, Qwen0.5B-chat)** (§4.4):
   `code/07_multi_arch_causal_patch.py` ->
   `results/multi_arch_causal_patch.json`

-  **Component-specificity test with a real Attention direction** (§4.4):
   `kaggle_kernels/paper1-causal-patch-real-attn-direction/` ->
   `results/causal_patch_real_attn_direction_results.json`

-  **Permutation cosine null, common-site test, random-direction ensemble**
   (§4.4): `kaggle_kernels/paper1-causal-patch-tier1-validated/` ->
   `kaggle_kernels/paper1-causal-patch-tier1-validated/output/causal_patch_tier1_validated_results.json`

-  **TOST equivalence bounds, competing-risks table** (§4.4):
   `code/38_tier1_tost_competing_risks.py` ->
   `results/tier1_tost_competing_risks.json`

-  **Low-dose alpha sweep** (§4.4): `code/42_low_dose_alpha_sweep.py` ->
   `results/low_dose_alpha_sweep.json`

-  **FFN/Attn dosage diagnostic** (§4.4):
   `code/26_ffn_attn_dosage_diagnostic.py` ->
   `results/ffn_attn_dosage_diagnostic.json`

-  **FFN vs. Attention component decomposition** (§4.5, §4.6):
   `code/02_cross_arch_component_probe.py` ->
   `results/cross_arch_component_probe_*.json`

-  **Paired DeltaAUROC and nested-CV selection-bias check** (§4.5):
   `code/37_paired_component_delta_auroc.py` ->
   `results/paired_component_delta_auroc.json`

-  **Qwen chat-template reversal** (§4.6):
   `code/02_cross_arch_component_probe.py qwen05chat` ->
   `results/cross_arch_component_probe_qwen05chat.json`

-  **ROME-style causal tracing** (Appendix D):
   `code/08_rome_style_causal_tracing.py`,
   `code/09_multi_arch_rome_style_causal_tracing.py`,
   `code/18_rome_style_causal_tracing_scaled.py`,
   `code/40_rome_style_causal_tracing_validated.py` ->
   `results/rome_style_causal_tracing*.json`

-  **SAE feature clamp (GPT-2 null; companion positive control)** (Appendix
   D): generating script for the GPT-2 null not released (predates current
   numbering); `code/15_sae_feature_gating_utility.py` ->
   `results/sae_feature_clamp_paper1.json`,
   `results/sae_feature_gating_utility.json`

-  **Difficulty-matched and adversarial controls** (Appendix E):
   `code/06_difficulty_matched_control.py`,
   `code/11_multi_arch_difficulty_matched_control.py`,
   `code/17_gradient_reversal_adversarial_probe.py`,
   `code/30_difficulty_matched_control_judge_label.py` ->
   `results/difficulty_matched_control*.json`,
   `results/multi_arch_difficulty_matched_control.json`,
   `results/gradient_reversal_adversarial_probe.json`

-  **Layer localization (7 methods), Jaccard and validated label** (Appendix
   F): `code/00_verify_vendored_mechint_numbers.py`,
   `code/29_gpt2_full_validated_relabel_rerun.py` ->
   `results/vendored_mech_int/`,
   `results/gpt2_full_validated_relabel_rerun.json`

-  **Surface-feature baseline under the Jaccard label (the 6 features, and the
   0.531/0.576 figures)** (Appendix E, §6): `code/05_run_surface_baseline.py`,
   `code/05_surface_baseline_classifier.py` -> `results/surface_baseline/`

-  **Label-validity audit and surface baselines** (§6):
   `code/16_llm_judge_label_noise.py`,
   `code/23_regenerate_completions_for_judge.py`,
   `code/24_llm_judge_score_all_architectures.py` (run as
   `kaggle_kernels/paper1-llm-judge-relabel/`),
   `code/32_surface_baseline_vs_judge_label.py`, `code/41_cheap_baselines.py`
   -> `results/llm_judge_label_noise.json`,
   `results/llm_judge_relabel_summary.json`,
   `results/surface_baseline_vs_judge_label.json`,
   `results/cheap_baselines.json`

-  **GPT-2 full-534 judge relabel** (§6):
   `kaggle_kernels/paper1-gpt2-full-judge-relabel/`,
   `code/28_judge_label_all_gpt2_534.py` ->
   `results/gpt2_full_534_judge_labels.json`

-  **Figure 1** (§4.6): `code/20_generate_ffn_attn_figure.py` ->
   `draft/latex/figures/ffn-attn-comparison.pdf` (plots already-reported
   numbers; no new computation) itemize

## References

Author lists, titles and dates were re-verified against each work's arXiv or
ACL Anthology record for this revision; an earlier version listed five entries
by title alone because no author list had been recorded in this project, and
one entry carried an unverified venue ("ParamMute, NeurIPS 2025"), corrected
to a preprint citation here.

Sun, Y., et al. (2025). ReDeEP: Detecting Hallucination in Retrieval-Augmented
Generation via Mechanistic Interpretability. *ICLR 2025*. arXiv 2410.11414.

Huang, P., Liu, Z., Yan, Y., Zhao, H., Yi, X., Chen, H., Liu, Z., Sun, M.,
Xiao, T., Yu, G., & Xiong, C. (2025). ParamMute: Suppressing
Knowledge-Critical FFNs for Faithful Retrieval-Augmented Generation. arXiv
2502.15543.

Wang, L. (2025). SEReDeEP: Hallucination Detection in Retrieval-Augmented
Models via Semantic Entropy and Context-Parameter Fusion. arXiv 2505.07528.

Dassen, M., Kotula, R., Murray, K., Yates, A., Lawrie, D., Kayi, E., Mayfield,
J., & Duh, K. (2026). FACTUM: Mechanistic Detection of Citation Hallucination
in Long-Form RAG. arXiv 2601.05866.

Xiong, G., He, Z., Liu, B., Sinha, S., & Zhang, A. (2025). Toward Faithful
Retrieval-Augmented Generation with Sparse Autoencoders. arXiv 2512.08892.

Roy, D., Misra, R., Singh, S. K., & Roy, A. (2026). Detection Without
Correction: A Robust Asymmetry in Activation-Based Hallucination Probing.
arXiv 2604.13068.

Meng, K., Bau, D., Andonian, A., & Belinkov, Y. (2022). Locating and Editing
Factual Associations in GPT. *NeurIPS 2022*. arXiv 2202.05262.

Meng, K., Sharma, A. S., Andonian, A., Belinkov, Y., & Bau, D. (2023).
Mass-Editing Memory in a Transformer. *ICLR 2023*. arXiv 2210.07229.

Dai, D., Dong, L., Hao, Y., Sui, Z., Chang, B., & Wei, F. (2022). Knowledge
Neurons in Pretrained Transformers. *ACL 2022*. arXiv 2104.08696.

O'Neill, C., Chalnev, S., Zhao, C. C., Kirkby, M., & Jayasekara, M. (2025). A
Single Direction of Truth: An Observer Model's Linear Residual Probe Exposes
and Steers Contextual Hallucinations. arXiv 2507.23221.

Kantamneni, S., Engels, J., Rajamanoharan, S., Tegmark, M., & Nanda, N.
(2025). Are Sparse Autoencoders Useful? A Case Study in Sparse Probing. arXiv
2502.16681.

Hewitt, J., & Liang, P. (2019). Designing and Interpreting Probes with Control
Tasks. *Proceedings of EMNLP-IJCNLP 2019*, 2733--2743.

Janiak, D., Binkowski, J., Sawczyn, A., Gabrys, B., Shwartz-Ziv, R., &
Kajdanowicz, T. (2025). The Illusion of Progress: Re-evaluating Hallucination
Detection in LLMs. *EMNLP 2025*. arXiv 2508.08285.

Hussain, K., & Kantarcioglu, M. (2026). PARALLAX: Separating Genuine
Hallucination Detection from Benchmark Construction Artifacts. arXiv
2605.17028.

Li, K., Patel, O., Viégas, F., Pfister, H., & Wattenberg, M. (2023).
Inference-Time Intervention: Eliciting Truthful Answers from a Language Model.
*NeurIPS 2023*. arXiv 2306.03341.

Kuhn, L., Gal, Y., & Farquhar, S. (2023). Semantic Uncertainty: Linguistic
Invariances for Uncertainty Estimation in Natural Language Generation. *ICLR
2023*. arXiv 2302.09664.

Manakul, P., Liusie, A., & Gales, M. (2023). SelfCheckGPT: Zero-Resource
Black-Box Hallucination Detection for Generative Large Language Models. *EMNLP
2023*. arXiv 2303.08896.
