# FFN Over-Retrieval Does Not Cleanly Extend to Closed-Book Confabulation

**Lakshmi Chakradhar Vijayarao**
Independent Researcher
`lakshmichakradhar.v@gmail.com`

## Abstract

ReDeEP (Sun et al., ICLR 2025) established a "knowledge FFN" mechanism:
feed-forward sublayers can override retrieved context during
hallucination in retrieval-augmented generation. We test whether the
same asymmetry appears in pure closed-book generation, across three
architectures spanning a ~4.0x parameter range (GPT-2, Pythia-410M,
Qwen2.5-0.5B-Instruct).

FFN beats Attention as a passive hallucination-detection signal on a
numerical majority of layers on all three architectures, but no
per-architecture test is significant, and every peak-AUROC margin sits
within cross-validation noise. One architecture's result reverses
entirely once a template artifact (bare prompt vs. chat template) is
fixed -- evidence against a scale story. Two difficulty-matched controls
and an adversarial probe rule out question-difficulty as a generic
confound, though inconsistently across architectures. **A further, more
severe problem, found by a subsequent independent review: every
passive-probe AUROC in this paper is computed under standard random
K-fold cross-validation on TruthfulQA, which never checks for
category-clustering leakage (questions cluster into 38 topics whose
correct-answer rates range 0%-10.5%, so a probe can learn topic instead
of hallucination). A leave-one-category-out re-test of the flagship
component probe (GPT-2, L8/L9) collapses AUROC from 0.62-0.66 under
standard CV to 0.48-0.49 -- chance or below -- at every cell tested.
Extending this check to the other two architectures gives a
heterogeneous result, not a uniform one: Pythia-410M's component probe
survives leave-one-category-out CV essentially unchanged at both tested
layers (no detectable contamination), while Qwen2.5-0.5B's partially
collapses at both components (0.566->0.485 FFN, 0.563->0.526 Attention --
the numerically larger drop for FFN is not itself tested for
significance and should not be read as an established
component-specific asymmetry). Category leakage is therefore a real,
substantial, but architecture- and component-dependent contaminant of
this paper's passive-probe evidence, not a uniform artifact explaining
away all three architectures' results at once, and the other five
converging methods have not yet been checked.**

A direct causal test -- patching the FFN sublayer during generation,
against a genuinely Attention-derived control direction, under an
independently validated label -- shows zero measurable FFN-specificity.
Five further checks (a held-out direction-validity gate, a
permutation-based cosine null, a common-injection-site test, a
random-direction ensemble, and two alternative direction estimators)
converge on the same conclusion from a different angle: neither
direction was ever shown to carry validated causal signal in the first
place, so this null is a limitation of the instrument, not evidence the
two components are equally causal. We tried to grow the held-out set by
judging 283 previously-unused prompts; zero were judged correct,
confirming GPT-2's near-zero success rate on this task is a genuine
competence ceiling, not a sampling artifact -- the causal test still
replicates cleanly at nearly double the power. A stronger causal method
(ROME-style tracing) finds no effect surviving correction under the
validated label. A sparse-feature intervention finds no
passively-associated feature at
all.

The ReDeEP mechanism does not cleanly extend to closed-book
confabulation at this scale -- and the passive-probing evidence that
originally motivated testing it may not have been established on a
leakage-free protocol in the first place. This paper's most transferable
contribution is methodological: a six-question validity checklist for
causal-patching studies, a leave-one-category-out re-test protocol for
passive hidden-state probes on category-structured benchmarks like
TruthfulQA, and a construct-validity check that catches over half of
baseline "hallucinated" completions as degenerate repetition loops rather
than confident false claims -- disciplines reusable well beyond this
specific mechanism.

## 1. Introduction

Fluent text and factual correctness are distinct properties of language
model output. ReDeEP (Sun et al., ICLR 2025) localized one mechanism by
which they diverge in retrieval-augmented generation: FFN sublayers
overriding retrieved context, a "Knowledge FFN" over-retrieval pattern.
Does the same pattern operate in pure closed-book generation, where there
is no retrieved context to override? To our knowledge no ReDeEP-lineage
follow-up has tested this. We test it directly on GPT-2, replicate on
Pythia-410M and Qwen2.5-0.5B-Instruct, and probe it causally via targeted
FFN-sublayer activation patching.

We use **closed-book FFN over-retrieval** as a precise, testable label for
this question, not a new mechanism we claim to discover: does an FFN
sublayer's activity carry a measurably stronger, more direct relationship
to hallucination than an Attention sublayer's, when there is no retrieved
context for either component to override? Two independent signatures
would jointly constitute a positive result: (a) a *passive* signature --
FFN-derived hidden-state features detect hallucination with reliably
higher AUROC than Attention-derived features at the two components'
respective peak layers (§3.2-3.3); and (b) an *active* signature --
patching an FFN-derived direction into the FFN sublayer during generation
corrects more hallucinations than an equivalent, genuinely
Attention-derived direction patched into the Attention sublayer (§3.4).
Neither signature alone is conclusive on its own terms: a passive AUROC
gap could still reflect a correlate rather than a cause, and a causal
null could still reflect an underpowered or mis-specified direction
rather than a true absence of effect. That is why this paper reports
both, plus two further causal probes (§3.5-3.6) and two confound controls
(§3.7), rather than treating either signature alone as decisive.

The number of methods applied to what is ultimately one question is
deliberate: a passive layer-majority signal this small (a few points of
AUROC) admits several mundane alternative explanations -- an uncontrolled
prompt-template confound, a question-difficulty confound, a
construct-invalid outcome metric, an underpowered causal test -- and each
additional method rules out exactly one of these before the remaining
result is taken at face value. That several of them ruled the effect
out, rather than confirming it, is the finding.

Why should a reader outside this specific mechanism's lineage care? Not
because the paper discovers a new mechanism -- it doesn't. The reason is
methodological and general: before this paper's causal-patching
experiment can speak to "does FFN activity cause hallucination," it must
survive the question "is the underlying flip-to-correct metric even
measuring semantic correction, or is over half of it repetition-breaking
on degenerate completions?" (§3.4 finds the latter, for 51.9-53.1\% of
baseline "hallucinated" completions). Any researcher running a causal
intervention on an LLM and reporting a flip-rate metric inherits this
same risk, independent of which component or behavior is being tested.
That transferable discipline, not the specific FFN finding, is this
paper's most durable contribution.

**Contributions, honestly scoped:**
1. Localization of hallucination signal to layers 8-9 on GPT-2 under the
   Jaccard label (6 converging methods, triangulating rigor rather than
   a qualitatively new discovery -- mid-layer hallucination localization
   is established prior art). A seventh method (DLA magnitude) peaks at
   L10/L11 instead and is excluded from this count. **This convergence
   itself does not survive relabeling**: under the validated judge label,
   the four of these methods rerun on the full 534-sample relabel (dense
   probe, sparse L1 probe, token-position probe, and DLA) move to L7, L7,
   L6, and L11 respectively (§3.1) -- L8-9 is a property of the Jaccard
   label's specific noise pattern on this dataset, not a label-independent
   localization, and
   this contribution should be read accordingly.
2. FFN-vs-Attention component decomposition testing whether ReDeEP's
   RAG-scoped mechanism extends to closed-book QA: a numerical FFN
   majority on 3/3 architectures under a bare-template first pass, with
   every per-architecture binomial test non-significant and the
   layer-pooled test invalid (adjacent layers are not independent
   trials) (§3.2-3.3).
3. Identification of an instruction-tuning/template confound on Qwen0.5B
   and its correction: rerunning with the proper chat template flips
   both of Qwen0.5B's component-comparison metrics to Attention-favoring
   under the Jaccard label, changing the honest architecture count to
   2/3 -- though re-probing under a validated label (§3.3) restores
   FFN's majority, so this reversal is itself label-sensitive.
4. A targeted FFN-sublayer causal-patching experiment showing zero
   measurable FFN-vs-Attention specificity, using a genuinely
   Attention-derived direction (nearly orthogonal to the FFN direction)
   as the actual component-specificity control -- McNemar
   $p=0.08$-$1.00$ across four tested configurations at $n=467$ under an
   independently validated label, none significant (§3.4).

## 2. Related Work

**Parametric factual knowledge in FFN sublayers.** ROME (Meng et al.
2022) and MEMIT (Meng et al. 2023) causally locate and edit specific
facts by writing to mid-layer FFN weight matrices; Knowledge Neurons (Dai
et al. 2022) independently identifies FFN-resident neurons whose
activation correlates with specific factual recall. Our finding is a
much weaker, aggregate, correlational claim than theirs: these works show
FFN sublayers are *causally editable* loci of individual facts, whereas
we find only that FFN sublayer activity *as a group* is a
modestly-better-than-chance hallucination signal *on average*. We do not
conflate the two.

**RAG-scoped FFN over-retrieval.** ReDeEP (Sun et al., ICLR 2025)
established FFN override of retrieved context during RAG hallucination.
Follow-up work (ParamMute, NeurIPS 2025; SEReDeEP, arXiv 2505.07528;
FACTUM, arXiv 2601.05866; RAGLens, arXiv 2512.08892) refines this
mechanism or extends it with sparse-autoencoder tooling, but all remain
confirmed RAG-only and do not test the closed-book question this paper
addresses.

**Detection without correction.** An independent finding (arXiv
2604.13068) reports that output-confidence baselines beat activation
probes above ~410M parameters, and that residual-stream steering flips
0/7 tested models' generated answers toward correct across 42
configurations on GPT-2-scale models. We compare our probe against a
matched 6-feature surface baseline (0.605 vs.\ 0.576,
`code/05_run_surface_baseline.py`, MLP 5-fold CV AUROC
$=0.5756\pm0.0684$) and find a directionally consistent, weak margin. Our
FFN-sublayer causal-patching experiment (§3.4) directly tests whether a
component-targeted intervention does better than their generic
residual-stream result; it does not (McNemar $p=0.08$-$1.00$, none
significant) -- an independent replication of their null at the
component level.

**Black-box detection.** Semantic entropy (Kuhn et al. 2023) and
SelfCheckGPT (Manakul et al. 2023) operate purely on output text and
generalize to closed API-only models where hidden states are
unavailable -- the actual production-relevant baseline this paper's
mechanistic probes would need to beat. We make no such comparison here
and flag its absence explicitly.

**Sparsity and steering method provenance.** The 100/768-dimension sparse
probe result (§3.1) is consistent with Kantamneni et al. ("Are Sparse
Autoencoders Useful? A Case Study in Sparse Probing," arXiv 2502.16681),
which finds dense L1 probes often match or beat trained SAE probes; it is
not evidence for a claim made by the superficially similar "A Single
Direction of Truth" (arXiv 2507.23221), which does not make this
comparison. The difference-of-means steering method used throughout
descends from Inference-Time Intervention (Li et al. 2023, arXiv
2306.03341).

**A recurring meta-pattern.** This paper's own confound-hunting -- the
chat-template reversal (§3.3), the difficulty-matched control that turns
out to test a weaker-than-assumed confound (§3.7) -- is not an isolated
case. Related, independently-conducted work in adjacent areas of LLM
interpretability and evaluation has repeatedly found that an initially
plausible signal or severity estimate substantially weakens or changes
character once a specific, previously-unexamined confound is directly
tested, rather than merely acknowledged as possible. Passive linear and
geometric probes on LLM hidden states are pervasively vulnerable to
superficial confounds that are easy to overlook and consistently shrink
or eliminate the apparent signal once tested directly.

## 3. Methods and Results

### 3.1 Layer localization (GPT-2, 6 converging methods, plus a corrected 7th)

All seven numbers below are reproducible directly from this repo
(vendored at `results/vendored_mech_int/`, verified byte-identical to the
unshipped mech-int project's own logs where an in-project copy already
existed; `code/00_verify_vendored_mechint_numbers.py` reprints each for
direct verification).

Six methods converge on layers 8-9: dense probe peak L9 (0.5827); sparse
L1 probe peak L9 (100/768 active dims, 87\% sparse, CV AUROC 0.589, not
the inflated in-sample 0.874); token-position probe peak L8 last-token
(0.6036); component probe FFN peak L8 (0.6053) vs.\ Attn peak L3
(0.6165); steering peak improvement at L9; gold-token logit lens
divergence at L8. A seventh method, DLA magnitude, does not join this
convergence: L9 does not have the largest absolute DLA in the underlying
data (L10 $+0.90$ and L11 $+0.71$ are larger), so L8-9 is supported by
the other six methods, not DLA magnitude.

Two of the six are weaker than "converges on L8-9" implies. Steering's
"peak improvement" at L9 is $0.0015$ AUROC points, the argmax over a
52-cell grid against per-layer CV standard deviations of $0.03$-$0.08$ --
not distinguishable from argmax-over-noise. The logit-lens analysis
computes two divergence-layer estimates from the same run: the plain
correct-vs-hallucinated divergence peaks at layer 1, not layer 8; only
the gold-token variant peaks at L8, which is the one reported above.

**Re-checking layer localization under the validated label.** Since §4
shows the Jaccard label agrees with an independent LLM judge only
52\% of the time on GPT-2 ($\kappa=0.04$), we reran the four methods that
consume only cached activations -- dense probe, sparse L1 probe,
token-position probe, and DLA -- with the same full-534-sample judge
label used throughout this validation effort
(`code/29_gpt2_full_validated_relabel_rerun.py`, reusing the vendored
mech-int probing code unmodified against the same cached
`activations.pkl`, only the label array swapped). The peak layer shifts
under the validated label rather than staying at L8-9: dense probe peaks
at L7 ($0.6444$, up from L9's $0.5827$); sparse L1 probe also peaks at L7,
now far sparser (19/768 active dims, 98\% sparse, vs.\ 100/768, 87\%
sparse under Jaccard), CV AUROC $0.6260$; token-position probe peaks at
L6 last-token ($0.6961$, up from L8's $0.6036$); DLA's largest FFN
difference moves from L10 ($+0.90$) to L11 ($+1.81$). Absolute AUROCs
rise under the validated label here exactly as they do for the other
three architectures in §3.2, but the peak layer itself is not stable
across labels -- L8-9 is a property of the Jaccard label's specific
noise pattern, not a label-independent fact about where GPT-2 localizes
this signal. **This localization result must be read together with the
class-imbalance caveat in §3.2**: the validated label leaves only 27/534
(5.1\%) of GPT-2 samples "correct," and cross-validated AUROC at this
imbalance is substantially noisier (e.g.\ the token-position probe's peak
SD is not separately reported here, but see §3.2's paired FFN/Attn SDs
for the same 27/507 split, which roughly double under the validated
label) -- so the specific new peak layers above should be read as
suggestive of instability in the Jaccard-based localization, not as a
newly-established, precise alternative localization.

### 3.2 FFN vs. Attention component decomposition (GPT-2)

On GPT-2, FFN wins 8/12 layers (two-sided binomial $p=0.39$; one-sided
$p=0.19$; neither significant). Peak FFN layer is L8 (AUROC 0.6053); peak
Attn layer is L3 (AUROC 0.6165). **The single best-discriminating
component on GPT-2 is Attention, not FFN.** At L8, FFN direct logit
attribution is higher for hallucinated samples (5.08) than correct
samples (4.85) -- an in-sample, non-cross-validated, suggestive but
unconfirmed "over-retrieval" signature.

**A category-leakage check the standard CV protocol never ran.** A
fresh, independent adversarial review raised the single highest-priority
missing check in this paper: TruthfulQA questions cluster into 38 topical
categories (Misconceptions, Law, Health, Fiction, ...), and standard
random K-fold CV -- the protocol every probe number in §3.1-3.2 uses --
can place same-category, near-duplicate questions in both a fold's train
and test split. Category is not independent of the hallucination label
either: on this same 534-item GPT-2 pool, among the 10 most frequent
categories (each n>=15), correct-answer rate ranges from 0% (Fiction,
Paranormal, Stereotypes) to 10.5% (History, Conspiracies); across all 38
categories including smaller ones the range is wider still (0% to 28.6%
at Confusion: People, n=7 -- full per-category breakdown, including
per-category n, in `results/category_leakage_diagnostic.json`'s
`category_correct_rates` field), so a probe that learns to recognize
*topic* could achieve an above-chance AUROC with zero genuine
hallucination-related content.

We ran this check for the component probe, GPT-2, L8 and L9. Using the
identical judge-labeled pool and last-token FFN/Attn extraction as the
causal-patching kernels (`code/47_category_leakage_diagnostic.py`), we
compare the standard 5-fold CV AUROC against a leave-one-category-out CV
(train on 37 categories, test on the 1 held out, repeated for every
category with both classes represented in its held-out slice -- 16 of 38
categories qualify; the remaining 22 have zero correct-labeled items and
are skipped, a direct consequence of the same 27/534 class imbalance
disclosed above). The result is unambiguous and severe: standard CV gives
AUROC 0.622 (L8 FFN), 0.632 (L8 Attn), 0.616 (L9 FFN), 0.663 (L9 Attn) --
closely matching the vendored numbers §3.1 reports above (L8 FFN 0.6053)
-- while leave-one-category-out CV collapses every one of these to
chance or below: 0.479, 0.491, 0.482, 0.489 respectively (SD 0.27-0.35
across the 16 held-out categories). **This is exactly the failure mode
that would close the passive-probing line permanently**: the standard CV
protocol's above-chance AUROC for the component probe does not survive
removing category-clustering leakage, at either tested layer, for either
component.

**Extending this check to Pythia-410M and Qwen2.5-0.5B-Instruct: a
heterogeneous, not a uniform, result.** We reran the identical
generation/labeling/extraction methodology (Jaccard word-overlap
labeling, mean-pooled per-layer FFN/Attention activations, bare
"Q: {question}\nA:" prompts) on the full 817-item TruthfulQA validation
split for both remaining architectures, extracting only at each
architecture's own already-reported peak FFN and peak Attn layer
(Pythia: L11/L4; Qwen0.5B: L8/L17;
`kaggle_kernels/paper1-category-leakage-cross-arch/`) and comparing
standard 5-fold CV against leave-one-category-out CV exactly as above.
Unlike GPT-2's severely imbalanced pool (27/534 correct, 5.1%, which left
only 16 of 38 categories usable for LOGO-CV), both of these
architectures' Jaccard-labeled pools are well-balanced (Pythia: 282/605,
46.6% correct, 37/38 categories usable; Qwen0.5B: 253/513, 49.3% correct,
34/38 usable) -- a materially more reliable LOGO-CV estimate than GPT-2's.
The result does not replicate GPT-2's collapse uniformly: **Pythia's
component probe survives leave-one-category-out CV essentially
unchanged** (L11 FFN: standard 0.618 -> LOGO 0.617; L4 Attn:
0.612 -> 0.602), showing no detectable category-leakage contamination at
either tested layer. **Qwen0.5B partially collapses at both
components**: L8 FFN drops from 0.566 to 0.485 -- to
chance or slightly below, the same qualitative pattern GPT-2 showed --
while L17 Attn drops more modestly, from 0.563 to 0.526. The FFN drop is
numerically larger, but at 34 LOGO folds (SD 0.20/0.22) this difference
is only ~0.9 standard errors, not itself statistically tested; we
report both point estimates and do not claim a significant
component-specific asymmetry. **The honest,
now-complete picture across all three architectures is heterogeneous, not
a uniform artifact**: category-leakage contamination appears to explain
GPT-2's component-probe signal almost entirely, explains a substantial
part of Qwen0.5B's FFN signal specifically but only a modest part of its
Attention signal, and is not detectable in Pythia at all. This means the
broader claim below that "FFN wins a numerical majority of layers on all
three architectures" cannot be uniformly dismissed as category-leakage
artifact, but neither can it be trusted at face value for GPT-2 or (for
FFN specifically) Qwen0.5B -- each architecture's passive-probe result
now needs to be read on its own terms with respect to this confound, not
assumed to transfer from one to the others in either direction. We did
not (yet) rerun the other five converging methods in §3.1 (dense probe,
sparse probe, token-position probe, steering, logit lens) under
leave-one-category-out CV; given the heterogeneity just found across
architectures for the same method, we no longer assume a uniform answer
for those methods either, and flag this as the natural next check. Full
per-architecture, per-layer results:
`results/category_leakage_diagnostic.json` (GPT-2),
`results/category_leakage_cross_arch_results.json` (Pythia, Qwen0.5B).

### 3.3 Cross-architecture data (GPT-2, Pythia-410M, Qwen2.5-0.5B-Instruct)

[Full version: `draft/cross_architecture_section.md`; real Kaggle data,
$N=605$ Pythia / $N=513$ Qwen0.5B.] FFN wins a numerical majority of
layers on all three architectures (66.7\%, 66.7\%, 58.3\%) -- though
**§3.2's category-leakage re-test (now run for all three architectures)
shows this majority-of-layers signal is itself confounded**: largely
explained by category leakage for GPT-2, partially so for Qwen0.5B
(especially its FFN component), and not detectably confounded for
Pythia; per-architecture two-sided p-values are 0.39/0.15/0.54, one-sided
0.19/0.076/0.27 -- all non-significant. Pooled across all 60 layers,
38/60 FFN wins gives a nominally significant one-sided $p=0.026$, but
this is not a valid inferential instrument: 60 layers within only 3
architectures are strongly autocorrelated, not independent trials. The
only cleanly poolable count is architecture-level (3/3), directionally
consistent but far too small an $n$ to test formally.

On the paper's primary metric, peak AUROC, Attention is the single
best-discriminating component on **one** of three architectures (GPT-2),
not two. Pythia's peak AUROC favors FFN (L11$=0.6181$ vs.\ Attn
L4$=0.6115$); Qwen0.5B's also favors FFN (L8$=0.5657$ vs.\ Attn
L17$=0.5625$). Every one of these margins sits within a quarter to a
fifth of the relevant peak's own cross-validation SD -- GPT-2's
$0.011$ margin against SDs of $0.0557$/$0.0427$; Pythia's $0.0066$
margin against $0.0442$/$0.0345$. **The honest, uniform statement is: the
peak-component question is within measurement noise on all three
architectures tested, not just Qwen.**

**A paired test of the actual estimand, and a check for selection bias
in "peak AUROC" itself.** The comparisons above are visual: two
separately-estimated CV means checked against their own fold SDs, even
though the FFN and Attention probes are fit on the *same* samples and
folds -- they are paired, and the paper's real estimand is their
difference, $\Delta=\text{AUROC}_{\text{FFN}}-\text{AUROC}_{\text{Attn}}$,
which was never directly tested. Using already-cached raw per-sample
features (Pythia and Qwen0.5B) and GPT-2's vendored mech-int activations
(no new model inference for any architecture), we compute, for every
layer on every architecture, out-of-fold predicted probabilities for
both components from the same folds, and a BCa bootstrap 95\% CI on
$\Delta$ over 2000 resamples of the samples themselves
(`code/37_paired_component_delta_auroc.py`; this pools out-of-fold
predictions into one AUROC per component rather than averaging five
per-fold AUROCs, so the absolute point estimates below differ slightly
from the mean-of-folds convention used elsewhere in this paper -- the
CI is what is new here, not the point estimate). At each architecture's
own naive FFN-peak layer: GPT-2 (L8) $\Delta=+0.067$, CI $[+0.012,
+0.122]$ (excludes zero -- though L8 is exactly the cell §3.2's
leave-one-category-out re-test shows collapsing from AUROC 0.62 to 0.48,
so this CI-excludes-zero result should not be read as established signal
independent of that confound); Pythia (L11) $\Delta=+0.047$, CI $[-0.002,
+0.100]$; Qwen0.5B (L20) $\Delta=+0.053$, CI $[-0.007,+0.113]$. **Note a
naming collision, disclosed here rather than left implicit:** this
pooled-OOF aggregation's own argmax gives Qwen0.5B's FFN peak as L20 and
Attn peak as L8 -- the reverse of the mean-of-folds peaks reported
elsewhere in this paper (FFN L8, Attn L17, §3.3): L8 is "the FFN peak"
under one aggregation convention and "the Attn peak" under the other for
the same architecture, a consequence of pooled-OOF and mean-of-folds
AUROC estimation genuinely disagreeing on which layer wins, not a typo.
At each architecture's own naive Attn-peak layer (by this section's own
pooled-OOF convention), the sign reverses as expected
(Attn winning at its own peak): GPT-2 (L3) $\Delta=-0.085$, CI
$[-0.137,-0.036]$; Pythia (L4) $\Delta=-0.113$, CI $[-0.162,-0.064]$;
Qwen0.5B (L8) $\Delta=-0.032$, CI $[-0.084,+0.023]$. Averaged across
*all* layers rather than only the peaks -- the summary least
vulnerable to selection bias -- $\Delta$ is small on every architecture
(GPT-2 $+0.0017\pm0.036$; Pythia $+0.0005\pm0.042$; Qwen0.5B
$-0.0021\pm0.034$), and a layer-weighted pooled estimate across all
three architectures gives $\Delta=-0.0003$ with between-architecture
variance of only $3.9\times10^{-6}$ -- as close to a formal, properly
pooled null result as this paper's data supports, replacing the earlier
informal "3/3 architectures, too small to test."

We also checked "peak AUROC" itself for winner's-curse-style selection
bias, using nested cross-validation (select the peak layer on inner
folds only, evaluate on a held-out outer fold). Nested-CV peaks are
uniformly lower than the naive argmax peaks reported above -- GPT-2 FFN
$0.643\to0.581$, GPT-2 Attn $0.632\to0.600$; Pythia FFN
$0.632\to0.584$, Pythia Attn $0.666\to0.582$; Qwen0.5B FFN
$0.563\to0.521$, Qwen0.5B Attn $0.570\to0.515$ -- confirming the
selection bias this comparison is structurally exposed to. On GPT-2,
this correction is large enough to flip which component is ahead: the
naive comparison has FFN leading ($0.643$ vs.\ $0.632$), but the
nested-CV-corrected comparison has Attention leading instead ($0.600$
vs.\ $0.581$). We do not read this as evidence Attention "really" wins
on GPT-2 -- the corrected margin is itself well within noise -- but it
is a concrete demonstration that the peak-AUROC comparison this section
otherwise relies on can select for the noisier, not the truer, of the
two components on a given architecture.

Qwen2.5-0.5B (~494M parameters, the largest model tested, not the
smallest) shows the weakest and most equivocal signal, and was queried
with a bare `Q: ... A:` template rather than its chat template --
genuinely out-of-distribution usage for an instruction-tuned model, and a
more parsimonious explanation for its weak/reversed results than any
story about scale. **Rerunning with the proper chat template**
(`code/02_cross_arch_component_probe.py qwen05chat`; only the prompt
construction changed) **reverses the peak-component result**: Attention
becomes the clear peak (L4$=0.5988\pm0.0438$) over FFN (L4$=0.5704\pm0.0186$,
same layer), and the layer-count result reverses too (FFN wins 11/24,
45.8\%, a minority, versus 58.3\% before). Both reversals point the same
direction: the bare-template run's FFN-favoring numbers were an artifact
of the OOD confound, not a weaker version of the same signal. The
corrected margin ($0.0284$) is still smaller than Attention's own CV SD
($0.0438$), so "Attention now clearly wins" would overclaim in the same
way "FFN wins 3/3" originally did -- the finding is the direction of the
flip, not a newly-resolved winner. Peak-FFN depth fraction also shifts,
from 33.3\% (L8/24, bare template) to 16.7\% (L4/24, chat template) --
one more respect in which the original numbers described an artifact of
prompting, not a property of the model.

**Re-probing under a validated label.** Given the word-overlap label's
validity concerns (§4), we relabeled every completion on all three
architectures with the same LLM judge used throughout this paper and
reran this component probe under both labels for direct comparison
(`code/24_llm_judge_score_all_architectures.py`,
`results/llm_judge_relabel_summary.json`), and separately extended the
same check to GPT-2 itself
(`code/29_gpt2_full_validated_relabel_rerun.py`, all 534 samples).
Absolute AUROCs rise substantially under the validated label on every
architecture (GPT-2 $0.605$/$0.617\to0.698$/$0.717$ FFN/Attn peak;
Pythia $0.615\to0.735$/$0.749$; Qwen0.5B-bare $0.566\to0.713$/$0.699$;
Qwen0.5B-chat $0.570$/$0.599\to0.660$/$0.641$), and FFN's numerical
majority is restored on Qwen0.5B-chat specifically (11/24 layers under
Jaccard, a minority, to 18/24 under the validated label, a majority) --
the template-reversal finding above is thus itself sensitive to which
label is used. **The full layer-majority picture, stated completely
rather than selectively:** under the validated label, FFN's numerical
majority actually *rises* on three of four architectures (Pythia
$16/24\to18/24$; Qwen0.5B-bare $14/24\to20/24$; Qwen0.5B-chat
$11/24\to18/24$) and only GPT-2 moves the other way, reversing further
toward Attention ($8/12\to2/12$). At the architecture level, using the
same four-way tally under both labels, this leaves FFN holding a
layer-count majority on 3 of 4 architectures under the validated label --
unchanged in count from Jaccard's own 3 of 4 (GPT-2, Pythia,
Qwen0.5B-bare), but with different member architectures: GPT-2 flips
out of FFN's majority as Qwen0.5B-chat flips in. Neither a clean
strengthening nor a clean reversal of the original picture.

Which component peaks is, if anything, *less* stable across labels than
the layer-count tally: only 2 of 4 architectures keep the same
peak-AUROC winner under both labels (GPT-2: Attention both times;
Qwen0.5B-bare: FFN both times), while the other 2 flip in opposite
directions (Pythia: FFN under Jaccard $\to$ Attention under the
validated label; Qwen0.5B-chat: Attention under Jaccard $\to$ FFN under
the validated label). An earlier version of this paragraph described
Pythia as "Attention still peaks" and both Qwen0.5B variants as "FFN
still peaks" -- both wrong: "still" implies no change, and two of the
four architectures' peak component changes label to label. This
re-check strengthens the case that a real, larger signal than the noisy
label suggested exists, without resolving the paper's core
FFN-vs-Attention specificity question either way -- and if anything
underscores how unstable the specific peak-component assignment is.

**A class-imbalance caveat that applies to every validated-label number
above.** The judge label is not just different from Jaccard, it is
severely imbalanced toward "hallucinated": GPT-2 27/534 correct (5.1\%),
Pythia 29/605 (4.8\%), Qwen0.5B-bare 63/513 (12.3\%), Qwen0.5B-chat
73/433 (16.9\%) -- confirmed directly from each architecture's judge
confusion matrix. Cross-validated AUROC at this level of imbalance is
substantially noisier: on GPT-2, the FFN/Attn peak AUROC standard
deviations roughly double under the validated label relative to Jaccard
(FFN $0.0557\to0.1115$; Attn $0.0427\to0.1253$), so GPT-2's own
$0.698$/$0.717$ margin ($0.019$) is well within one SD of either peak,
not a more confident finding than the Jaccard-label near-tie it
reverses. The rising absolute AUROCs and the layer-majority reversals
above are real properties of the validated-label re-analysis, not
artifacts of a coding error, but they should be read as suggestive of a
real, currently under-characterized effect of label quality on this
probe, not as tighter estimates than the Jaccard-label numbers they
revise -- a larger, class-balanced validated-label sample would be needed
to state the post-correction peak margins with the same confidence the
original Jaccard-label numbers (wrongly) conveyed.

\begin{figure}[h]
\centering
\includegraphics[width=0.75\textwidth]{figures/ffn-attn-comparison.pdf}
\caption{Peak AUROC for FFN vs.\ Attention across every tested condition, with error bars showing each peak's own cross-validation standard deviation. The margin between components is within one CV~SD of overlap in all four conditions, and the one architecture that initially showed a clearer FFN edge (Qwen0.5B, bare template) reverses to favor Attention once queried with its proper chat template.}
\label{fig:ffn-attn-comparison}
\end{figure}

### 3.4 Causal verification: FFN-sublayer patching (GPT-2, Pythia, Qwen)

[Real data: `results/ffn_causal_patch_results.json`,
`results/ffn_causal_patch_scaled_results.json`.] A difference-of-means
"truthfulness direction," computed on FFN sublayer output (train split
only), is injected additively into the FFN sublayer during generation,
tested at L8/L9, $\alpha\in\{20,40\}$, against a random-direction control
and an attention-sublayer-patch control.

**Construct validity, checked before trusting the outcome metric.** We
scanned baseline "hallucinated" completions for a repeated 4-8 word
phrase occurring 3+ times. On the original 70/30-split test set ($n=81$):
42/81 (51.9\%, Wilson 95\% CI [41.1\%, 62.4\%]) are degenerate repetition
loops, not confident false statements. Re-checked at the dataset's
maximum $n=228$ (a leaner 15/85 split, `code/10_ffn_causal_patch_scaled.py`):
121/228 (53.1\%, CI [46.6\%, 59.4\%]) -- a stable property of GPT-2's
TruthfulQA failure mode, not a small-sample artifact. This means a large
additive intervention "flipping" the output to a reference-matching
string is, for roughly half this test set, at least as consistent with
breaking a repetition loop as with correcting a hallucinated claim.

**Result, at adequate power -- for the FFN-vs-random comparison only.**
At the maximum supportable $n=228$ (27-46 discordant pairs per
configuration -- enough that a real, moderate effect should be visible
if one existed), all four FFN-found-vs-FFN-random McNemar tests are
decisively non-significant ($p=0.522, 0.868, 1.000, 0.659$ at
L8/$\alpha{=}20$, L8/$\alpha{=}40$, L9/$\alpha{=}20$, L9/$\alpha{=}40$).
Restricting further to the 107 prompts confirmed non-degenerate, the
null holds if anything more uniformly ($p=1.000, 1.000, 1.000, 0.503$).
This comparison's scope is FFN-found-vs-FFN-random only: the
"Attention" condition in this same script patches the FFN-derived
direction at the Attention sublayer's site, not a direction derived from
Attention's own activations, so it cannot support an FFN-vs-Attention
component-specificity claim (Appendix B). The genuine version of that
test, with a real Attention-derived direction, is reported below.

A non-trivial fraction of interventions in every condition produce a
degenerate/unparseable completion rather than a clean correct-or-wrong
answer, ranging $10.1\%$-$46.5\%$ across all twelve found/random/
attn-found conditions at both layers and alphas
(`results/ffn_causal_patch_scaled_results.json`). Found and random
directions are at broadly similar rates in most configurations but
diverge sharply at L8/$\alpha{=}40$ (FFN-found $40.8\%$ vs.\ FFN-random
$25.4\%$) -- the found direction is, if anything, a \emph{stronger}
disruptor of coherent generation than a random direction of equal norm
at this configuration, consistent with much of the flip-rate signal
being generic generation perturbation, not targeted semantic correction.
Because unparseable completions are scored as not-flipped in every
condition, this differential degeneration could mechanically penalize
whichever arm degenerates more; we do not attempt a competing-risks
correction here and flag this as an unresolved limitation of the outcome
metric.

**Extending beyond GPT-2.** The same test on Pythia-410M and
Qwen2.5-0.5B-Instruct (chat-templated), at each architecture's own
established peak layer, is too underpowered to support any claim. On
Pythia, valid pairs collapse from $n=22$ at $\alpha=10$ ($p=0.25$) to
$n=7$ at $\alpha=20$ to $n=0$ at $\alpha=40$ (generation degenerates
entirely). On Qwen0.5B-chat, valid pairs are $n=2,1,0$ at
$\alpha=10,20,40$ -- uninformative rather than merely underpowered, since
this instruction-tuned model's chat-style responses rarely clear the
word-overlap labeling threshold even at baseline. Neither extension
supports or contradicts the GPT-2 finding; Pythia's pattern is consistent
with GPT-2's, and Qwen0.5B-chat's failure is a labeling-threshold
mismatch, not evidence about the causal question itself.

**Validating the null against the label itself.** The most serious
objection to every result above is that the Jaccard label defining both
the test set and the flip-to-correct outcome is unreliable (§4 quantifies
this at $\kappa=0.04$ on GPT-2). We addressed this directly rather than
leaving it as a caveat: an independent LLM judge relabeled all 534 of
GPT-2's original completions ($\kappa=0.03$ against Jaccard, consistent
with the GPT-2-only audit), and we reran the causal-patching
protocol -- found-direction computed from a judge-labeled train split
($n=18$ judge-correct, $n=40$ judge-hallucinated -- the small correct-class
count is a direct consequence of the severe class imbalance disclosed in
§3.2, and is itself a caveat on how stable this direction estimate can
be), patched generation, every output scored by the same judge -- on all
467 judge-hallucinated test prompts (nearly double the $n=228$ used
above). FFN-found-vs-FFN-random flip-to-correct rates collapse to
1.3-3.0\% at every layer/alpha under the validated label, with no
configuration distinguishable from its random-direction control
(McNemar $p\geq0.52$ throughout).

**The corrected FFN-vs-Attention component-specificity test.** The
comparison this section previously reported as "McNemar $p=1.000$ in
every configuration" for FFN-found vs.\ Attention-found was never
actually a test of Attention specificity (previous paragraph, and
Appendix discussion above): the "Attention" arm always injected the
FFN-derived direction at the Attention sublayer's site, not a direction
derived from Attention's own activations. We fixed this by computing a
genuine Attention-derived direction -- identical difference-of-means
methodology to the FFN direction, just extracting last-token Attention
sublayer output instead of FFN output, from the same judge-labeled train
split -- and rerunning the full protocol with this real direction injected
at the Attention site
(`kaggle_kernels/paper1-causal-patch-real-attn-direction/`,
`results/causal_patch_real_attn_direction_results.json`). The two
directions are nearly orthogonal (cosine similarity $-0.054$ at L8,
$-0.056$ at L9), confirming they are genuinely different directions, not
an accidental near-duplicate. On this actually-valid test, at the same
$n=467$ judge-hallucinated prompts: FFN-found vs.\ Attention-found gives
McNemar $p=0.167$ (L8/$\alpha{=}20$), $0.607$ (L8/$\alpha{=}40$), $1.000$
(L9/$\alpha{=}20$), $0.077$ (L9/$\alpha{=}40$) -- not significant at any
configuration, and the smallest (L9/$\alpha{=}40$, $p=0.077$) would not
survive even a mild multiple-comparison correction across the four
configurations tested.

**What this test can and cannot rule out, stated explicitly.** Discordant
pairs per configuration are $19$ (L8/$\alpha{=}20$), $15$ (L8/$\alpha{=}40$),
$9$ (L9/$\alpha{=}20$), $16$ (L9/$\alpha{=}40$) -- fewer than the
$27$-$46$ available at $n=228$ under the Jaccard label, because the much
lower absolute flip rate under the validated label leaves fewer prompts
where FFN-found and Attention-found disagree at all. At these counts, the
minimum odds ratio an exact two-sided binomial test (McNemar) can detect
at $p<0.05$ is $3.75$ (L8/$\alpha{=}20$), $4.00$ (L8/$\alpha{=}40$),
$8.00$ (L9/$\alpha{=}20$), and $4.33$ (L9/$\alpha{=}40$) -- this test is
powered only to detect a 4-to-8-fold asymmetry between the two
conditions, not a moderate one. The observed odds ratios and their
approximate 95\% confidence intervals (log-odds normal approximation)
are $0.46$ $[0.18, 1.21]$, $0.67$ $[0.24, 1.87]$, $1.25$ $[0.34, 4.65]$,
and $3.00$ $[0.97, 9.30]$ -- wide enough that the data are simultaneously
consistent with FFN being several-fold worse than Attention and with FFN
being several-fold better, at three of the four configurations. This is
the honest scope of "no measurable difference": it is a demonstrated
absence of a very large effect, not a demonstrated absence of a moderate
one, and we did not run an equivalence test (e.g.\ TOST against a
pre-specified $\delta$) that would let us make the stronger claim.
Attention-found is also indistinguishable from
its own random-direction control at every configuration ($p=0.84, 0.50,
1.00, 0.18$), just as FFN-found is from FFN-random. The same test
rerun under the original Jaccard label (same prompts and patches, scored
by the cheaper heuristic instead) gives FFN-vs-Attention $p=0.439,
0.355, 0.747, 0.399$ -- also uniformly non-significant.

**Each arm against its own random-direction control.** Of the eight
found-vs-random tests this run computes (both components, both layers,
both alphas), three are nominally significant: Attention-found is beaten
by its own random control at L8 ($p=0.021$, $\alpha{=}20$; $p=0.0043$,
$\alpha{=}40$), and FFN-found beats its own random control at
L9/$\alpha{=}20$ ($p=0.0264$). Only the L8/$\alpha{=}40$ Attention
comparison ($p=0.0043$) survives Holm-Bonferroni correction across all
eight tests (threshold $0.05/8=0.00625$); the L8/$\alpha{=}20$ Attention
comparison ($p=0.021$) does not. An Attention direction performing
\emph{worse} than a random one is not evidence of Attention specificity
-- it is at least as consistent with generic large-perturbation
disruption (above) as with any targeted effect. The FFN result does not
survive correction and does not replicate under the validated judge
label at the same configuration ($p=1.000$), consistent with Jaccard
noise rather than a real effect.

**The null this
paper reports -- no measurable causal difference between patching FFN
and patching Attention at the 4-to-8-fold odds-ratio scale this test is
powered to detect -- holds up under every test the corrected run
computes, under both labels, not only the ones most favorable to that
conclusion.** This is a materially different, more precisely scoped basis
for the same conclusion the uncorrected version of this test claimed to
support. But a further round of validation (below) found that this
instrument's genuinely Attention-derived direction, like the FFN
direction, was never shown to carry validated causal-relevant content in
the first place -- so the honest reading of this null is narrower than
"we tested a genuinely component-specific intervention on each side and
found no difference": it is closer to "this specific instrument does not
clear the bar needed to make its own null claim informative about
component specificity." Appendix-A-equivalent detail on exactly what was
wrong with the earlier (same-direction) version of this test, and why,
is documented in this section rather than relegated to an appendix,
since it bears directly on this paper's single most load-bearing claim.

**The judge-parser fix (§4, Appendix B) checked directly on this specific
test, not just on the underlying relabel.** Because this specific test
scores 467 test prompts times seventeen conditions (baseline plus
sixteen patched variants) with the judge, we reran it end to end with
the corrected parser as a direct check, not an inference from the
relabel result (Appendix B). The output is byte-identical to the
pre-fix version at every one of the $467\times17$ scored cells and every
summary statistic reported above.

We do not carry forward the previous version's interpretive claim about
the gap between Jaccard's and the judge's absolute flip rates (33-42\%
vs.\ 1.3-3.0\%) reflecting "superficial word overlap" specifically: 226
of these 467 judge-hallucinated test prompts (48.4\%) were already
Jaccard-labeled *correct* at baseline, before any patching, so a
substantial share of Jaccard's higher "flip-to-correct" rate reflects
prompts that did not need to flip at all, not a cleanly interpretable
gap between the two labels' sensitivity to superficial correctness. The
judge-labeled test set, by construction (selected as judge-hallucinated),
does not have this problem, and is the version of this comparison we
rely on. Full methodology and per-configuration numbers:
`kaggle_kernels/paper1-causal-patch-judge-label/` (original,
non-component-specific test, retained for the FFN-found-vs-FFN-random
result and superseded for the FFN-vs-Attention comparison),
`kaggle_kernels/paper1-causal-patch-real-attn-direction/` (the corrected
component-specificity test), `results/causal_patch_judge_label_results.json`,
`results/causal_patch_real_attn_direction_results.json`.

**A dosage-mismatch check on the FFN-vs-Attention comparison itself
(Appendix B documents two errors an earlier version of this diagnostic
made and how they were fixed).** The same $\alpha$ is added to both
sublayers' output in every configuration above; whether this is a fair
comparison depends on what that $\alpha$ is actually competing with. The
quantity the intervention actually competes with, at the point it
enters the computation, is the residual stream's own norm at that
layer, which both the FFN-site and Attention-site hooks add into almost
identically (both accumulate the same embedding-plus-all-prior-layers
history, differing only by one attention-sublayer contribution at layer
8-9, small relative to eight-plus layers of accumulation). Measured
correctly (`code/26_ffn_attn_dosage_diagnostic.py`, $n=100$
correctly-formatted prompts): residual-stream norm is $124.7$ (L8) and
$166.0$ (L9), giving a relative perturbation of $16.0\%$/$12.0\%$ of the
residual stream at $\alpha=20$ and $32.1\%$/$24.1\%$ at $\alpha=40$ --
\emph{identical for both the FFN-site and Attention-site interventions},
not the $2$-$3\times$ asymmetry the uncorrected version reported (which
also shrinks to $1.3$-$1.5\times$ on the correct prompts alone, before
even fixing the denominator). There is therefore no dosage asymmetry
between the two arms to explain the observed FFN-vs-Attention
non-difference away in either direction; both intervention sites receive
the same relative push into the residual stream. This does not by
itself validate the FFN-vs-Attention causal test's design -- the
Attention arm's more serious problem, that it never used a genuinely
Attention-derived direction, is addressed next -- but it rules out
dosage imbalance specifically as a confound.

**Eight further checks, run on the flagship configuration
(`kaggle_kernels/paper1-causal-patch-tier1-validated/`), a fresh audit
identified as missing.** These checks do not simply reinforce the null
reported above; they surface a more fundamental limitation of this
specific instrument that the paper must disclose honestly rather than
narrate past.

*Direction validity -- the central finding of this round.* **Disclosure,
made explicit here for the first time:** every direction and every
validity-AUROC measurement in this paper is computed from the model's
last-token activation on the *prompt alone* (`"Q: {question}\nA:"`),
never on any token of GPT-2's actual generated completion. The
judge/Jaccard correct-vs-hallucinated label used to supervise the
direction is a property of that separately-generated completion, but the
completion's own tokens are never passed through the model for this
measurement -- the direction is asked to separate, from the model's
internal state *before it has generated anything*, prompts that will go
on to receive a correct answer from prompts that will not. This is a
coherent question (does the model's pre-generation state predict its own
upcoming failure?), but it is a substantially harder and different
question than probing completion tokens for features of an
already-hallucinated answer, and a reader should not assume otherwise
from "last-token activation" language alone. Before any
direction is injected, does it actually separate correct from
hallucinated activations on genuinely held-out data? We split the
58-prompt training pool into a direction-fit set (47) and a held-out
validity set (11: 3 correct, 8 hallucinated), estimated each direction on
the former, and measured its scalar projection's AUROC on the latter. At
neither tested layer does either direction clear chance in the helpful
direction (L8 FFN/Attn AUROC$=0.083$/$0.083$, bootstrap 95\% CI
$[0.0,0.333]$ both; L9 FFN AUROC$=0.0$, CI $[0.0,0.0]$; L9 Attn
AUROC$=0.125$, CI $[0.0,0.5]$). **Correction:** an earlier draft
described all four CIs as "consistent with chance"; three of the four in
fact exclude $0.5$ (chance) entirely, and an exact Mann-Whitney test at
this $n_{\text{pos}}=3$, $n_{\text{neg}}=8$ split finds them nominally
significant in the *anti-predictive* direction (L8 FFN/Attn $p=0.0242$
each, L9 FFN $p=0.0121$; only L9 Attn, $p=0.0848$, is not significant).
Under a Bonferroni correction across these four cells ($\alpha=0.0125$),
only L9 FFN survives.

A further correction to the power analysis itself matters here: an
earlier draft's Monte Carlo power criterion was one-sided (CI-lower-bound
$>0.5$ only), structurally blind to power in the anti-predictive
direction the observed data actually falls in. Recomputed with a
two-sided criterion (CI excludes $0.5$ on either side), this test at
$n=11$ is asymmetric but reasonably well-powered exactly where the
observed AUROCs land: $80\%$ power is reached at a true AUROC of $0.05$
in the anti-predictive direction (versus $0.95$ in the helpful
direction), and power at the true AUROCs closest to the observed point
estimates is substantial (e.g. $73\%$ at a true AUROC of $0.10$, $89\%$
at $0.05$). This means the nominally significant anti-predictive
p-values above are *not* simply underpowered noise -- this test can and
does detect deviations of this size reliably. What it cannot resolve, at
a single realization with $n=11$, is whether this specific sample's
anti-predictive ranking reflects a real property of these directions on
this validity split, or a chance draw from a null distribution that this
small, discrete sample space (only $\binom{11}{3}=165$ possible rank
assignments) makes coarse enough that "nominally significant" and "one
unlucky draw" are not yet distinguishable without a second, independent
validity split.

**A 200-resplit diagnostic resolves this.** Using the identical 534-item
judge-labeled pool and train-pool construction (58 items: 18 correct + 40
hallucinated, an 80/20 direction-fit/validity-holdout split per class)
but redrawing which specific items land in each role at 200 different
random seeds instead of the kernel's single seed, the held-out AUROC
varies substantially (mean $0.54$-$0.58$, SD $\approx0.20$; range
$[0.083,1.0]$ at L8 FFN/Attn, $[0.0,1.0]$ at L9 FFN/Attn) and is centered
at or slightly *above* chance, not below it. The original seed's anti-predictive AUROCs
sit in the extreme low tail of this distribution: only $1.5\%$ of
resplits (L8 FFN, L8 Attn), $0.5\%$ (L9 FFN), and $4\%$ (L9 Attn) of the
200 resplits produce an AUROC at or below what the kernel's single split
happened to draw. **This settles the question the exact Mann-Whitney
test above could not**: the original split's nominally significant
anti-predictive result is an atypical, unlucky single draw, not a stable
property of these directions on this pool -- the broader resampling
distribution is consistent with chance (if anything, mildly above it),
matching the causal-patching test's own independent finding (random-
direction ensemble, TOST, permutation cosine null, all below) that the
found directions carry no detectable signal in either direction. Full
per-resplit results: `results/direction_validity_resplit_diagnostic.json`
(`code/46_direction_validity_resplit_diagnostic.py`). Two
further checks confirm this is not an artifact of the estimator or the
sample size: logistic-regression weights and Fisher LDA, fit on the
identical split, do no better (all 12 layer/component/estimator
combinations at or below AUROC $0.167$); and the same power analysis
shows this test only reaches 80\% power in the helpful direction once
the true held-out AUROC is $\approx0.95$ -- confirming that direction
validity, specifically, remains impossible to confirm at any single
$n=11$ split, which is exactly why the resampling view above, not any
one split, is the trustworthy picture.

*Random-direction ensemble -- the strongest single check.* At the
flagship configuration (L8, $\alpha=40$, 60 test prompts), the found FFN
and Attention directions' flip rates were compared against 20
independently-drawn random directions on the identical prompts. The
found directions produced **zero flips** -- sitting at the **50th**
(FFN) and **40th percentile** (Attention) of the random-direction
distribution (range $0.0$-$0.067$). The found direction's causal effect
is not merely statistically indistinguishable from random -- it sits
squarely where a direction with no special relationship to the label
would be expected to land.

*Five further checks, summarized.* A permutation-based cosine null
(2000 draws) finds the FFN/Attention directions' cosine similarity
($-0.0058$/$-0.0051$; corrected from a $10\times$ transcription error in
an earlier draft) statistically indistinguishable from two
label-uninformed directions (both within 1 SD of the empirical null). A
common-injection-site test (patching both directions at the shared
post-block residual stream, disentangling "which direction" from "where
injected") confirms FFN-native-site and FFN-common-site are
mathematically identical for GPT-2 (verified byte-for-byte), and finds
the genuinely new Attention-common-site comparison non-significant at
every configuration (one nominal $p=0.049$ does not survive Holm
correction across even these four tests). A TOST equivalence search
finds no cell establishes equivalence at the pre-registered OR$=2.0$
bound; the smallest achievable bounds range $2.85$-$10.9$. A
competing-risks (flip/no-flip/degenerate) $2\times3$ chi-square finds
one nominal $p=0.010$ that does not survive correction either. A
low-dose sweep ($\alpha\in\{2.5,5,10\}$ at the common site, Jaccard
label for speed) finds flip rates flat across all three doses
($42$-$52\%$), no monotone dose-response. All five are consistent with,
and do not add beyond, the two findings above.

**Attempting to grow the held-out set: the bottleneck is GPT-2's
competence, not sample size.** The direction-validity gate's $n=11$
holdout is bottlenecked by GPT-2's judge-validated correct rate on
TruthfulQA (27/534, 5.1\%) -- itself a consequence of an old Jaccard-label
filter that discarded 283 of the full 817-item TruthfulQA validation
split before this judge existed. We judged all 283 previously-discarded
items and merged them into the pool
(`kaggle_kernels/paper1-causal-patch-enlarged-pool/`): \textbf{zero of
them were judged correct} (27/817 correct overall, exactly unchanged from
27/534) -- GPT-2's near-zero success rate on TruthfulQA is a genuine
competence ceiling, not an artifact of which 534 items happened to be
scorable by the old heuristic. The direction-validity holdout therefore
remains $n=11$; re-drawn from a differently-ordered pool, its AUROC point
estimates move (L8 FFN/Attn$=0.375$/$0.333$, L9 FFN/Attn$=0.167$/$0.25$,
all CIs still consistent with chance) -- a second independent
illustration of how much a single held-out AUROC estimate swings at this
$n$. The 200-resplit diagnostic above resolves what this single second
draw could only illustrate: across many resplits the held-out AUROC
centers near chance (mean $0.54$-$0.58$), and this second draw's point
estimates (0.167-0.375) sit well inside that typical range, unlike the
original seed's (0.0-0.125), which was an extreme low-tail draw.
The causal test itself reruns cleanly at nearly double the power
($n=750$ hallucinated prompts, up from 467): the null holds throughout
(FFN-vs-Attention common-site $p=0.18$-$0.84$ across all four
configurations), with minimum-detectable odds ratios of $2.5$-$8.0$
(previously $3.25$-$6.0$ at $n=467$) -- tighter at three configurations
(L8/$\alpha{=}20$: $3.25\to2.5$; L8/$\alpha{=}40$: $3.25\to2.625$;
L9/$\alpha{=}40$: $4.0\to3.0$), *looser* at the fourth (L9/$\alpha{=}20$:
$6.0\to8.0$, because the number of discordant pairs this specific
comparison produces actually fell from 14 to 9 at the larger $n$, not
simply grown with it) -- still not a qualitative change in what this
test can rule out, but a reminder that MDE depends on discordant-pair
count, not sample size directly. The
random-direction-ensemble check, rerun on this enlarged pool (a
different specific 60-prompt subset, since the underlying prompt
ordering shifted), now places the found directions at the 95th (FFN)
and 100th (Attention) percentile rather than the original run's 50th and
40th -- driven by 2-3 flip events out of 60, the kind of count this
check cannot stably resolve either way. We report both results rather
than the more favorable one: this specific check is itself too
underpowered at $n=60$ to trust in either direction, which is consistent
with, not a reversal of, this section's overall conclusion.

**What this means for the flagship result.** These checks do not
simply add further support to the existing FFN-vs-Attention null; taken
together, they show that neither direction in this test was ever
demonstrated to carry validated causal-relevant content in the first
place. A null result from an instrument that cannot be shown to measure
anything is not informative about component specificity one way or the
other -- it is uninterpretable between "FFN and Attention are equally
(un)causal for hallucination" and "we injected two vectors indistinguishable
from noise, so of course nothing differed." The common-site and TOST
results are consistent with the paper's existing null, and the achievable
equivalence bounds line up with the previously-reported power
calculation -- but the direction-validity and random-direction-ensemble
findings are the more consequential result of this round: this specific
causal test, as designed (an 18-vs-40-sample difference-of-means
direction), does not clear the bar needed to make its own null claim
meaningful, independent of anything about FFN or Attention specifically.
We report this as an honest limitation of the instrument, not a claim
that has been strengthened.

### 3.5 ROME-style causal tracing: a stronger causal test

Additive mean-shift steering (§3.4) is a comparatively weak causal
instrument. We replace it with causal tracing (Meng et al. 2022),
adapted to closed-book QA (corrupting the whole question span, since no
single clean "subject span" applies): a clean run scores a forced-choice
$\text{logit\_diff}$ between the correct and incorrect reference answer's
first token; a corrupted run adds Gaussian noise to the question-span
embeddings; a restoration sweep patches each (layer, component)
one at a time and records the normalized restoration score; a
specificity control repeats this with a mismatched example's activation
instead of the example's own (`code/08_rome_style_causal_tracing.py`).

**Result, at the maximum powered sample ($n_{\text{valid}}=67$, the
dataset's supportable maximum, pre-registering the joint 24-test
correction as primary).** FFN shows no specific restoration effect
anywhere. Attention's strongest candidate (L9, own$-$shuffled $=+0.151$)
does not survive the joint Holm-Bonferroni threshold ($p=0.012$ vs.\
$0.05/24=0.00208$) -- attenuated from the paper's original, lower-powered
$n=45$ pass, where the same cell was larger ($+0.214$, $p=0.0026$) and
had survived only a less conservative per-family (not joint) scoping.
Instead, MLP L9 -- an *anti-specific* result, where a
mismatched example's activation restores discrimination *better* than
the example's own -- clears the strict joint threshold
($p=0.00086$, own$-$shuffled $=-0.203$). More data did not vindicate the
one borderline Attention-favoring signal this test produced; it did not
reverse sign -- it stayed positive in both passes, shrinking from
$+0.214$ to $+0.151$ while losing significance under the stricter joint
correction. We read this as consistent with the original Attn L9
finding being noise around a lenient-scoping threshold, not a real effect
suppressed by low power, and it reinforces rather than complicates this
paper's overall causal null. We do not claim the anti-specific MLP L9
result is itself a stable finding rather than one more stopping point in
a noisy series.

**Extending beyond GPT-2.** Neither Pythia-410M nor Qwen2.5-0.5B shows
any layer or component surviving correction under either the joint or
per-family framing -- a clean null at both architectures
(`code/09_multi_arch_rome_style_causal_tracing.py`). Pythia's smallest
uncorrected $p=0.0645$ (Attn L20); Qwen's smallest is $p=0.0041$ (Attn
L15), numerically close to GPT-2's result but still short of Qwen's own
stricter 24-layer per-family threshold. The one causal signal this test
found anywhere (GPT-2 Attn L9, itself only surviving under a
non-default, per-family correction scoping) does not replicate in either
extension architecture.

**Rerun under the validated judge label -- a more structural dependence
on the Jaccard label than §3.4 had, closed by a fresh audit.** Every
result above filters candidates on Jaccard label$=1$ ("clean" examples).
§4 establishes that only $27/534$ GPT-2 samples are actually judge-correct
-- so most of the "clean" examples driving this section's own
self-declared *stronger* causal test were, on this paper's own validated
numbers, likely hallucinations. We reran the sweep filtering on the
validated judge label instead (`code/40_rome_style_causal_tracing_validated.py`),
averaging $10$ independent corruption draws per example (removing the
original's selection on a single noisy realization: the old
"clean$>$corrupted" filter is dropped entirely, and all candidates are
retained regardless of any one draw's degradation) and ensembling the
mismatched-donor control over $10$ random donors instead of one
deterministic ring-neighbor. The judge-correct pool is small --
$17$ usable candidates out of $27$ total judge-correct GPT-2 samples,
after requiring a parseable correct/incorrect answer pair -- and this is
disclosed as a genuine power limitation, not silently absorbed. Across
individual corruption draws, only $53.5\%$ show real degradation from
corruption (clean$>$corrupted), underscoring why averaging over draws,
rather than selecting on one, is the right correction. No cell survives
Holm-Bonferroni correction at either this test's own 24-comparison
family or a stricter $120$-test family spanning all three
architectures' equivalent sweeps; the two smallest uncorrected p-values
are both Attention layers (L10 $p=0.026$, L11 $p=0.023$, both
own$-$shuffled positive), consistent in direction with this section's
Jaccard-label result above but clearing neither correction threshold at
this sample size. The one-instance-name limitation this round does not
address, disclosed rather than silently left as-is: the forced-choice
score remains first-token log-odds, not a length-normalized
full-sequence log-probability -- implementing the latter would require a
materially larger redesign of the causal-tracing hook machinery than was
tractable alongside the label-validity and averaging fixes in this
round.

### 3.6 Toward active causal control: beyond a single linear direction

Section 3.4's intervention is a single dense linear direction injected at
fixed strength; at $n=81$-228 this design cannot distinguish "FFN resists
causal correction in principle" from "a dense direction is simply the
wrong intervention class." A dense mean-difference direction averages
over every latent factor that differs between conditions (topic, length,
confidence, and any genuine truthfulness signal); if a correction signal
exists but is carried by a small number of sparse, near-monosemantic
features, additive steering along the dense direction would dilute it.
Sparse autoencoders (SAEs) are designed to recover exactly this
structure.

**Proposed protocol.** Train an SAE on FFN sublayer output activations
(unit-norm decoder columns, to keep the $\ell_1$ sparsity penalty
well-posed rather than evadable by shrinking the code while inflating
decoder norm); score each feature for hallucination-relevance via a
Mann-Whitney $U$ test (robust to the zero-inflation typical of SAE
activations) followed by Benjamini-Hochberg FDR control at $q=0.05$
(applied to $p$-values, not effect size, since ranking by $|d_j|$ alone
does not control the false discovery rate); construct a reconstruction-space
clamp on the top FDR-surviving feature(s) and inject it during
generation; repeat the found-vs-random-feature and FFN-vs-Attention
specificity controls from §3.4. Training an SAE from scratch at adequate
scale was infeasible within this project's budget; the following
substitutes a pretrained one for that step only.

**What we ran, and what we found.** We substitute
`jbloom/GPT2-Small-SAEs-Reformatted`'s layer-8 SAE ($d_{\text{sae}}=24{,}576$,
trained on 300M tokens of OpenWebText) -- a disclosed mismatch on two
axes at once: wrong hookpoint (residual stream, not FFN-sublayer output)
and wrong training distribution (general text, not
TruthfulQA-hallucination-specific). **0 of 24,576 features survive FDR
on this paper's own 534-example dataset**
(`results/sae_feature_clamp_paper1.json`; the generating script for this
specific result predates the current numbered `code/` sequence and is
not separately released -- the cached result is the sole surviving
artifact), so
the causal clamp step was never reached. This is a genuine null one
stage earlier than §3.4's causal test, but bounded by instrument
mismatch, not a clean absence-of-mechanism result: either axis of
mismatch alone could explain zero surviving features without implying no
sparse FFN-specific signal exists. Running the identical procedure on a
companion paper's HaluEval dataset ($n=500$, same SAE, same layer) as a
positive-control check of the pipeline itself
(`code/15_sae_feature_gating_utility.py`) finds 331/24,576 features
survive FDR there (best $p=4.8\times10^{-11}$), confirming the GPT-2 null
above is not a feature-selection bug. Yet even with a passively
significant feature in hand, the causal clamp test on that dataset shows
no specificity at any strength ($p=1.000, 0.508, 1.000$ at
$\eta=10,20,40$) -- convergent with, not contradicting, this paper's
dense-direction causal null.

### 3.7 Dissociating difficulty from hallucination signal

The $\approx0.03$ AUROC margin over the surface-feature baseline
(0.605 vs.\ 0.576) is small enough that a generalized question-difficulty
signal remains a live alternative to a hallucination-specific one. We
test this two ways on GPT-2, matching correct/hallucinated groups within
10 quantile bins of (1) mean token-level output entropy and (2) the
out-of-fold predicted probability of a logistic regression over all 6
surface features, then re-probing the identical FFN/Attn component
comparison on each matched set (`code/06_difficulty_matched_control.py`).
Entropy-matching retains $n=492/534$; the stronger composite match
retains $n=462/534$. Neither proxy correlates significantly with
correctness before matching (entropy $r=0.045$, $p=0.295$; composite
$r=0.024$, $p=0.578$) -- this dataset never had a statistically
detectable difficulty confound for either proxy to remove, so this is
evidence of survival under a weak test, not proof of dissociation under a
strong one.

**Result: the FFN/Attention signal survives both matches essentially
unchanged**, including the stronger composite control (entropy-match:
FFN $0.6085\pm0.0442$, Attn $0.6102\pm0.0285$; composite-match: FFN
$0.6255\pm0.0510$, Attn $0.6253\pm0.0926$; vs.\ $0.6053\pm0.0557$ /
$0.6165\pm0.0427$ unmatched). A label-permutation test (500 shuffles,
replacing an invalid fold-std-as-SE z-test) gives both components
$p=0.0020$ under both matching schemes -- the permutation floor at this
shuffle count.

Extending to Pythia-410M and Qwen2.5-0.5B (fresh entropy computed from
each model's own greedy-decode logits;
`code/11_multi_arch_difficulty_matched_control.py`) splits by
architecture: Pythia replicates GPT-2's survival (FFN
$0.6327\pm0.0513$, Attn $0.6093\pm0.0221$, both $p=0.0020$, 89.3\%
retention), but Qwen0.5B does not (FFN $0.5277\pm0.0434$, $p=0.230$; Attn
$0.5282\pm0.0320$, $p=0.206$, 88.3\% retention) -- a genuine architecture
split, not a uniform confirmation.

**A stronger, adversarial version.** A gradient-reversal probe attaches a
second head predicting the entropy proxy from the same shared
representation, back-propagating its *negative* gradient -- actively
pushing the representation to discard whatever predicts difficulty while
still classifying hallucination
(`code/17_gradient_reversal_adversarial_probe.py`). The adversarial
pressure works as intended: held-out $R^2$ for the entropy head is
negative for both components ($-0.71$ FFN, $-1.44$ Attn), worse than
predicting the mean. Despite this, hallucination AUROC survives
significantly above chance for both (FFN $0.5944\pm0.0624$, Attn
$0.6134\pm0.0542$, both $p=0.0050$ against a 200-shuffle permutation
floor) -- the strictly stronger claim the matched-split control alone
could not make: the signal is not explained by difficulty leaking through
the representation, since it survives even when the representation is
adversarially trained to discard exactly that information.

**Attempting the same difficulty-matched control under the validated
label: inconclusive, not confirmatory or disconfirmatory.** We reran the
entropy- and composite-matched controls on GPT-2 with the same validated
judge label used elsewhere in this section
(`code/30_difficulty_matched_control_judge_label.py`). This control probes
the Jaccard-label peak layers (FFN L8, Attn L3) inherited unchanged from
§3.1-3.2's Jaccard analysis, not the validated label's own shifted peaks
(FFN L5, Attn L6) -- a layer mismatch we did not correct for, disclosed
here rather than left unremarked. The severe class
imbalance noted in §3.2 (only 27/534 judge-labeled correct) collides
directly with this control's per-bin matching procedure: with at most
1-4 correct samples per difficulty decile, the matched set retains only
54/534 samples (10.1\%, versus $\approx90\%$ under Jaccard above) --
27 per class, split five ways for cross-validation, roughly 5 per fold.
At this $n$, the two difficulty proxies give **inconsistent** answers: the
entropy-only match shows both components significantly above chance (FFN
$0.7840$, $p=0.0060$; Attn $0.7187$, $p=0.0200$ against a 500-shuffle
floor), while the stronger 6-feature composite match shows both at or
below chance and non-significant (FFN $0.3987$, $p=0.82$; Attn $0.3680$,
$p=0.89$). We do not read either result as informative about whether the
FFN/Attention signal survives a difficulty confound under the validated
label: a 5-per-fold CV probe is not powered to distinguish a real effect
from noise in either direction, and the retention collapse itself is a
direct, mechanical consequence of the label's class imbalance rather
than a property of the underlying difficulty-matching question. The
difficulty-dissociation result above should therefore be read as
established only under the Jaccard label; whether it holds under a
validated label remains genuinely untested, not weakly confirmed or
weakened, pending a larger validated-label sample with enough correct
completions to support matching.

## 4. Discussion and Limitations

**A validity checklist for causal-patching interpretability studies.**
§3.4's flagship causal test went through five rounds of construct-
validity scrutiny before its null could be trusted: a genuinely
component-specific control direction (not a reused, wrong-component
one), a dosage-match check, a direction-validity gate, a permutation-
based cosine null, and a random-direction ensemble -- and even after all
five, the direction-validity gate itself failed (neither the FFN nor
the Attention direction clears chance-level held-out AUROC at any
tested layer, §3.4). This paper's most transferable contribution is
this checklist, distilled from that process, for any study patching a
single estimated direction into a model's activations and drawing a
causal conclusion from the result:

1. **Is the estimated direction itself validated on held-out data
   before any intervention?** A direction that does not clear
   chance-level AUROC on a genuinely held-out split has not been shown
   to carry the signal an intervention experiment assumes it carries --
   patching it in and observing no effect is then uninformative about
   the underlying mechanism, not evidence against it.
2. **Is the "control" direction genuinely from a different source, not
   the same direction reused with a different label?** A control that
   is mathematically identical or near-identical to the treatment
   cannot establish specificity.
3. **Does the injection site confound "which representation" with
   "where in the computation"?** Patching two different sublayers at
   different points in the block entangles component identity with
   injection site; a common-site test disentangles them (§3.4) --
   and can reveal that two apparently different interventions are
   mathematically identical (as FFN-site and common-site patching are,
   for GPT-2's block structure, verified by direct algebra and an
   empirical byte-for-byte check).
4. **Is a null distinguishable from a genuinely random direction, not
   just from "no effect"?** A found direction whose causal effect sits
   inside the empirical distribution of random-direction effects (as
   this paper's does at the flagship configuration, 40th-50th percentile
   of a 20-draw ensemble -- though the enlarged-pool rerun places it at
   the 95th-100th percentile instead, and we report both rather than
   only the more favorable one) has not been shown to be doing anything
   a random direction would not also do.
5. **Is statistical power reported at the resolution the claim needs?**
   A null with a wide confidence interval (this paper's flagship test:
   minimum detectable odds ratio 3.75-8.0) rules out large effects, not
   small-to-moderate ones -- reporting the interval, not just the point
   estimate and p-value, is what lets a reader judge which claim the
   data actually support.
6. **Does an alternative direction-estimation method change the
   conclusion?** A single estimator's failure to validate is
   ambiguous between "this estimator is a poor choice" and "the effect
   is not there"; checking at least one alternative (this paper checks
   logistic-regression weights and Fisher LDA against the same
   held-out split, §3.4) helps distinguish the two.

Applying this checklist to this paper's own flagship result is why its
central claim is reported as a limitation of the instrument rather than
as confirmatory evidence for or against FFN-specific causal control.

**A second, distinct check for passive probing on category-structured
benchmarks.** The checklist above targets causal-patching studies
specifically. This paper's passive-probing results (§3.1-3.3) are
subject to a different, equally basic validity question that a fresh
review found this paper had never asked of itself: *does the probe's
above-chance CV AUROC survive removing the benchmark's own topical
category structure from the cross-validation split?* TruthfulQA clusters
into 38 categories whose correct-answer rates vary by over 10 percentage
points (§3.2), so a probe correlated with topic alone can look like a
hallucination detector under ordinary random K-fold CV. We ran a
leave-one-category-out re-test for the component probe on GPT-2 (§3.2):
standard CV AUROC of 0.62-0.66 collapses to 0.48-0.49 -- chance or below
-- at every layer/component tested. Extending this to Pythia-410M and
Qwen2.5-0.5B (§3.2) gives a heterogeneous result: Pythia's component
probe is essentially unaffected (no detectable category-leakage
contamination at either tested layer), while Qwen0.5B partially
collapses at both components (0.566->0.485 FFN, 0.563->0.526 Attn, an
untested numerical difference). Any study reporting a probe AUROC on TruthfulQA (or any
other benchmark whose items cluster into topics correlated with the
label) should run this check before treating a standard-CV AUROC as
evidence of the signal it is claimed to measure -- the result is neither
always contamination nor never contamination, so it cannot be assumed
either way without checking. We did not run this check for this paper's
other five converging methods (dense probe, sparse probe, token-position
probe, steering, logit lens) before this draft; this check is cheap (a
single re-partitioning of already-cached activations, no new model calls
for GPT-2, and a single additional generation pass per architecture
otherwise) and there was no principled reason it had been skipped until
an outside review asked for it.

**Data and code availability.** All code, cached result JSONs, and the
paper source are included in the anonymized supplementary material for
double-blind review, and will be released as a public GitHub repository
under the authors' name upon publication.
Every result that consumes only already-cached `results/*.json` artifacts
is self-contained. Three scripts additionally require an unshipped
sibling repository to *rerun from scratch* (as opposed to re-verifying
already-saved outputs): `code/01_ffn_causal_patch.py` imports live code
from a `mech-int` sibling project to regenerate labeled completions;
`code/03_fisher_geometry_ffn_attn.py` and
`code/06_difficulty_matched_control.py` depend on that same project's
2.9GB `activations.pkl`; `code/15_sae_feature_gating_utility.py` reads
cached hidden states from a second sibling project (`geom-proof`). Full
end-to-end reproduction of §3.4, §3.7, and the SAE-gating result from raw
data requires these two additional private repositories, not released
with this paper.

**Reproducibility map.**

| Claim | Section | Script | Cached result |
|---|---|---|---|
| Layer localization (7 methods) | §3.1 | `code/00_verify_vendored_mechint_numbers.py` | `results/*.json` (per-method) |
| FFN vs. Attention decomposition | §3.2 | `code/02_cross_arch_component_probe.py` | `results/cross_arch_component_probe_*.json` |
| Category-leakage re-test (LOGO-CV vs. standard CV), GPT-2 | §3.2 | `code/47_category_leakage_diagnostic.py` | `results/category_leakage_diagnostic.json` |
| Category-leakage re-test, Pythia-410M and Qwen2.5-0.5B | §3.2 | `kaggle_kernels/paper1-category-leakage-cross-arch/run_category_leakage_cross_arch.py` | `results/category_leakage_cross_arch_results.json` |
| Direction-validity 200-resplit diagnostic | §3.4 | `code/46_direction_validity_resplit_diagnostic.py` | `results/direction_validity_resplit_diagnostic.json` |
| Paired ΔAUROC, nested-CV selection-bias check | §3.3 | `code/37_paired_component_delta_auroc.py` | `results/paired_component_delta_auroc.json` |
| Qwen chat-template reversal | §3.3 | `code/02_cross_arch_component_probe.py qwen05chat` | `results/cross_arch_component_probe_qwen05chat.json` |
| Difficulty-matched control | §3.7 | `code/06_difficulty_matched_control.py`, `code/11_multi_arch_difficulty_matched_control.py` | `results/difficulty_matched_control.json`, `results/multi_arch_difficulty_matched_control.json` |
| FFN-sublayer causal patching | §3.4 | `code/01_ffn_causal_patch.py`, `code/10_ffn_causal_patch_scaled.py`, `code/14_causal_patch_scaled_degeneration_filter.py` | `results/ffn_causal_patch_results.json`, `results/ffn_causal_patch_scaled_results.json`, `results/ffn_causal_patch_scaled_degeneration_filtered.json` |
| ROME-style causal tracing | §3.5 | `code/08_rome_style_causal_tracing.py`, `code/09_multi_arch_rome_style_causal_tracing.py`, `code/18_rome_style_causal_tracing_scaled.py` | `results/rome_style_causal_tracing.json`, `results/multi_arch_rome_style_causal_tracing.json`, `results/rome_style_causal_tracing_scaled.json` |
| Adversarial gradient-reversal probe | §3.7 | `code/17_gradient_reversal_adversarial_probe.py` | `results/gradient_reversal_adversarial_probe.json` |
| SAE feature clamp, GPT-2 null (0/24,576 survive FDR) | §3.6 | generating script not released (predates current `code/` numbering) | `results/sae_feature_clamp_paper1.json`, `results/sae_feature_clamp_combined.json` |
| SAE feature clamp, companion-dataset positive control (331/24,576 survive FDR) | §3.6 | `code/15_sae_feature_gating_utility.py` | `results/sae_feature_gating_utility.json` |
| Label-validity audit (all 3 architectures) | §4 | `code/16_llm_judge_label_noise.py`, `code/23_regenerate_completions_for_judge.py`, `code/24_llm_judge_score_all_architectures.py` | `results/llm_judge_label_noise.json`, `results/llm_judge_relabel_summary.json` |
| Causal patching under validated label | §3.4 | `kaggle_kernels/paper1-causal-patch-judge-label/` | `results/causal_patch_judge_label_results.json` |
| Corrected FFN-vs-Attention component-specificity test (real Attention direction) | §3.4 | `kaggle_kernels/paper1-causal-patch-real-attn-direction/` | `results/causal_patch_real_attn_direction_results.json` |
| Direction-validity gate, permutation cosine null, common-site test, random-direction ensemble | §3.4 | `kaggle_kernels/paper1-causal-patch-tier1-validated/` | `kaggle_kernels/paper1-causal-patch-tier1-validated/output/causal_patch_tier1_validated_results.json` |
| TOST equivalence bounds, competing-risks outcome table | §3.4 | `code/38_tier1_tost_competing_risks.py` | `results/tier1_tost_competing_risks.json` |
| ROME-style causal tracing under validated judge label | §3.5 | `code/40_rome_style_causal_tracing_validated.py` | `results/rome_style_causal_tracing_validated.json` |
| FFN/Attn dosage-mismatch diagnostic | §3.4 | `code/26_ffn_attn_dosage_diagnostic.py` | `results/ffn_attn_dosage_diagnostic.json` |
| GPT-2 full-534 judge relabel | §3.1, §3.2, §3.7 | `kaggle_kernels/paper1-gpt2-full-judge-relabel/` | `results/gpt2_full_534_judge_labels.json` |
| Surface-feature (length/lexical) baseline vs. validated judge label | §4 | `code/32_surface_baseline_vs_judge_label.py` | `results/surface_baseline_vs_judge_label.json` |
| GPT-2 layer localization + component probe under validated label | §3.1, §3.2 | `code/29_gpt2_full_validated_relabel_rerun.py` | `results/gpt2_full_validated_relabel_rerun.json` |
| GPT-2 difficulty-matched control under validated label | §3.7 | `code/30_difficulty_matched_control_judge_label.py` | `results/difficulty_matched_control_judge_label.json` |
| Cheap baselines (last-layer probe, generation-confidence features) | §4 | `code/41_cheap_baselines.py` | `results/cheap_baselines.json` |
| Low-dose alpha sweep (common site, Jaccard label) | §3.4 | `code/42_low_dose_alpha_sweep.py` | `results/low_dose_alpha_sweep.json` |
| Direction-validity gate power/MDE analysis | §3.4 | `code/43_direction_validity_power_analysis.py` | `results/direction_validity_power_analysis.json` |
| Alternative direction estimators (logistic regression, LDA) | §3.4 | `code/44_alternative_direction_estimators.py` | `results/alternative_direction_estimators.json` |
| Enlarged-pool replication (283 previously-unused prompts judged, $n=750$ causal test) | §3.4 | `kaggle_kernels/paper1-causal-patch-enlarged-pool/` | `kaggle_kernels/paper1-causal-patch-enlarged-pool/output/causal_patch_enlarged_pool_results.json` |

**Label validity.** All results rest on a Jaccard word-overlap label,
surface-form divergence rather than verified factual incorrectness. We
quantified this on all three architectures with an independent LLM judge
(Qwen2.5-3B-Instruct): a 100-item stratified GPT-2 sample gives 52\% raw
agreement, Cohen's $\kappa=0.04$; full relabeling of every completion on
Pythia, Qwen0.5B-bare, and Qwen0.5B-chat gives $\kappa=0.032$, $0.141$,
and $0.084$ respectively -- next to chance throughout, not a GPT-2-specific
artifact. The disagreement is consistently biased in one direction: the
judge calls far more completions hallucinated than the word-overlap
heuristic does (GPT-2: 98\% agreement on Jaccard-hallucinated calls, 6\%
on Jaccard-correct calls), and manual reading of disagreement cases
confirms the judge is usually right -- word-overlap frequently credits
completions that share surface words with the reference but state a
different specific fact (e.g., naming the wrong person or film), or that
are degenerate repetition loops, as "correct." **A trivial length/lexical
baseline check (Appendix B notes why this computation needed to exist
before the claim below could be made).** We computed it
(`code/32_surface_baseline_vs_judge_label.py`, the identical 6-feature
surface classifier from §2/Related Work, rerun against the validated
label on the same cached features): logistic regression CV AUROC
$=0.5604\pm0.1047$, MLP $=0.5704\pm0.0971$ ($n=534$, 5-fold). This is
not chance-level, and it is not meaningfully different from the same
baseline's performance against the original Jaccard label
($0.531$/$0.576$) -- so the validated label is not obviously *easier*
for a surface-only classifier than the original label was. This is
weak evidence against the alternative that the validated-label rise in
hidden-state probe AUROC (§3.2-3.3, roughly $0.60\to0.70$) is driven
purely by degeneracy detection: a purely-degeneracy-driven account
would predict the surface baseline itself should also rise
substantially under the validated label, since entropy and generation
dynamics are the most direct surface signal of a repetition loop, and
it does not. This is not a strong exclusion -- both baselines remain
well short of the hidden-state probes' AUROC, and we did not restrict
either analysis to the confirmed-non-degenerate subset -- but it is
evidence, not merely an asserted fact with no computation behind it.

**Cheap baselines, on the same validated label.** Before attributing any
discriminative power to the FFN/Attention decomposition specifically, we
checked whether trivial generation-confidence signals do comparably well
(`code/41_cheap_baselines.py`, same $n=534$ validated label): an
undecomposed last-layer (final resid\_post) probe reaches AUROC
$=0.610\pm0.126$; teacher-forced generation-confidence features (mean
log-prob, min log-prob, mean/min per-token max-softmax) individually
reach $0.54$-$0.64$, with min-max-softmax the strongest single feature at
$0.643\pm0.065$, and all four combined at $0.635\pm0.077$. These sit in
the same range as this paper's own FFN/Attention component probes
($0.53$-$0.75$ depending on layer and label), meaning the specific
FFN-vs-Attention decomposition this paper studies does not obviously
outperform a simple, mechanism-agnostic confidence signal on this task --
consistent with this paper's overall finding that the component-level
mechanistic story is harder to establish than the raw AUROC numbers
alone would suggest.

Given this, we reran the paper's decisive causal test under
the validated label rather than leave the concern unresolved (§3.4): the
causal null holds within this test's power limits, corroborated by a
floor-level flip rate and zero separation between conditions at that
floor, though (as discussed in §3.4) the number of test prompts nearly
doubled while the number of statistically informative discordant pairs
actually fell. We extended this same validated-label relabeling
to all 534 of GPT-2's completions (not just the 100-item audit sample,
$\kappa=0.042$, consistent) and reran layer localization, component
decomposition, and the difficulty-matched control under it as well
(§3.1, §3.2, §3.7); the validated label there is severely imbalanced
toward "hallucinated" (27/534 correct, 5.1\%), which we flag explicitly
wherever it affects how much weight a resulting number should carry.
This paper's "correct" label should still be read
as "cleared a word-overlap threshold," not as "an independent judge
would also call this correct" -- but the paper's central causal claim no
longer depends on that distinction mattering. One residual limitation:
the same judge model both defines the found-direction's train split and
scores every generated output in §3.4's validated-label test, so that
result's validity rests on the judge's own accuracy, which we have
checked only by manual spot-reading and the surface-feature control
above (not chance-level, but not meaningfully different from the same
baseline's performance under the original label either), not an
independent second judge or human annotation.

**A parser fix, checked and confirmed to have zero effect on this
paper's reported labels (Appendix B).** All seven judge-scoring
implementations now check for `"HALLUCINAT"` and `"INCORRECT"` before
treating a bare `"CORRECT"` substring match as label $1$. Rerunning the
full 534-sample GPT-2 relabel
(`kaggle_kernels/paper1-gpt2-full-judge-relabel/`) end to end with the
corrected parser gives a result byte-identical to the original: all 534
labels unchanged, same $27$ correct / $507$ hallucinated split, same
$\kappa=0.0417$. We reran the §3.4 flagship causal test with the same
corrected parser as a further check, given how directly that result
depends on judge scoring; see §3.4 for the outcome.

**No inference-economy claim.** This paper localizes a signal and tests
a causal intervention; it does not propose an early-exit, routing, or
compute-saving mechanism, and none of its AUROCs (0.53-0.62) are strong
enough to gate anything at usable precision. We tested this directly on
the one passively-significant signal this project produced (the SAE
feature from §3.6's positive control, $p=4.8\times10^{-11}$):
thresholding it as a single-feature classifier reaches AUROC$=0.5614$,
but the feature's raw activation is at or near zero for nearly every
sample: no threshold above zero clears even 50\% recall at any of the
four target-recall operating points tested, so the PR-curve search
collapses to "predict everything positive," giving precision pinned at
the 4.8\% base rate throughout (`code/15_sae_feature_gating_utility.py`)
-- extreme statistical significance under simultaneous testing does not
translate into any usable gating concentration.

## 5. Conclusion

Whether ReDeEP's RAG-scoped FFN mechanism extends to closed-book
generation remains an open question after this study, not a confirmed
finding. FFN shows a directionally consistent but individually
non-significant numerical majority across three architectures under a
bare-template first pass; the layer-pooled test is not statistically
valid regardless of one- or two-sided scoring, due to within-architecture
autocorrelation. Correcting the Qwen0.5B chat-template confound (§3.3)
drops the architecture-level count to 2/3 and flips the single
best-discriminating component to Attention on two of three architectures
under the Jaccard label -- though re-probing under the validated label
(§3.3) restores FFN's majority on Qwen0.5B-chat specifically, so this
particular reversal should be read as label-sensitive, not settled.
A direct causal test, using a genuinely Attention-derived direction
(nearly orthogonal to the FFN direction) as the actual
component-specificity control rather than the FFN direction injected at
a different site, shows no measurable FFN-specificity (McNemar
$p=0.08$-$1.00$ across four tested configurations, none significant),
extending rather than contradicting an independent finding
that activation interventions fail to causally correct hallucinated
answers at this model scale. This null does not depend on trusting the
paper's word-overlap label: relabeling every completion with an
independent LLM judge and rerunning the causal test end to end -- on
nearly double the test prompts, though with fewer statistically
informative discordant pairs than the Jaccard-label version, since the
much lower absolute flip rate leaves fewer prompts where conditions
disagree -- leaves the null intact within this test's power limits
(minimum detectable odds ratio 3.75-8.0; §3.4 reports the exact
confidence intervals), corroborated by both a
floor-level flip rate and zero separation between conditions at that
floor. This is the paper's most methodologically scrutinized result --
the one shown to survive the paper's own most
serious methodological objection, both the label-validity concern and a
genuine (rather than fabricated) Attention-specific control -- but its
power to exclude a moderate effect remains limited, and we do not claim
more than that.

We report this candidly as a modest, largely null-leaning contribution.
Closed-book FFN over-retrieval (§1's operational definition: a passive
AUROC signature plus an active causal-patching signature) is a plausible
but empirically unconfirmed extension of ReDeEP -- neither signature was
observed reliably on any architecture tested. The paper's main value is
in what it honestly rules out -- clean FFN-specific causal control, a
clean scale story -- rather than what it positively establishes.

## Appendix B: Correction History

**This appendix documents bugs found and fixed during iterative review,
and checks run to confirm each fix's actual effect on reported numbers.**
§3.4 and §4 state the paper's results directly; nothing below is
required to verify them.

**A fabricated-looking claim, removed.** An earlier version of §3.4
claimed "a direct FFN-found-vs-Attention-found comparison gives McNemar
$p=1.000$ in every configuration." This was never actually computed:
`code/01_ffn_causal_patch.py` and its scaled variants save an
Attention-found flip rate but never run a McNemar test against it, and
in any case that condition patches the FFN-derived direction at the
Attention site rather than a direction derived from Attention's own
activations, so no version of this script could have supported an
FFN-vs-Attention specificity claim. The genuine test (§3.4) uses a real
Attention-derived direction instead. A related paragraph's degeneracy
rate ("19.8-29.6\% across conditions") was similarly only the FFN-found
arm at L8 from an earlier, smaller pass, not the full range across all
twelve conditions it was presented as describing; the correct range
($10.1\%$-$46.5\%$) is now what §3.4 reports.

**Judge-parser substring bug (checked directly against the flagship
causal test, not just the underlying relabel).** Every judge-scoring
function in this project originally classified a verdict as "correct"
if the string `"CORRECT"` appeared in it and `"HALLUCINAT"` did not --
`"CORRECT"` is a substring of `"INCORRECT"`, so a judge output of
"INCORRECT" would have been silently scored as label $1$ (correct) in
seven separate implementations. We had never persisted the judge's raw
verdict strings, only the parsed integer labels, so this was
unfalsifiable from existing artifacts until a fresh review flagged it.
Fixed by checking for `"HALLUCINAT"` and `"INCORRECT"` before treating a
bare `"CORRECT"` as label $1$, in all seven implementations, then reran
the full 534-sample GPT-2 relabel end to end with the corrected parser:
byte-identical to the original (same $27$/$507$ split, same
$\kappa=0.0417$). We additionally reran the §3.4 flagship causal test
(467 test prompts $\times$ 17 conditions) with the corrected parser as a
direct check, given how directly that specific result depends on judge
scoring, rather than inferring it was unaffected from the relabel result
alone: byte-identical at every one of the $467\times17$ scored cells.
For this judge model and this dataset, Qwen2.5-3B-Instruct reliably
complied with the one-word instruction and never actually produced a
verdict the buggy parser would have mishandled -- the bug was real and
needed fixing, but it did not change any number this paper reports.

**Dosage-mismatch diagnostic, corrected after a review caught two
errors in an earlier version.** An earlier version of
`code/26_ffn_attn_dosage_diagnostic.py` (1) measured on bare TruthfulQA
questions rather than the "Q: \{question\}\textbackslash nA:" formatted
prompts the causal test actually patches, and (2) normalized $\alpha$
against each sublayer's own raw output norm -- the wrong denominator,
since `patched_generate`'s hook replaces that output with
$(\text{out}+\alpha\cdot\text{direction})$ and this combined value is
then added into the residual stream by the transformer block's own
forward code, so the quantity the intervention actually competes with
is the residual stream's own norm at that layer, not the sublayer's own
output norm. The uncorrected version reported a $2$-$3\times$ dosage
asymmetry between the FFN and Attention arms; correcting only the
prompt format (keeping the wrong denominator) shrinks this to
$1.3$-$1.5\times$; correcting both gives the $16.0\%/12.0\%$
($\alpha=20$) and $32.1\%/24.1\%$ ($\alpha=40$) relative-perturbation
numbers reported in §3.4, with no asymmetry between the two arms.

**Surface-baseline computation, flagged as missing by a fresh review.**
An earlier version of §4's label-validity discussion asserted that a
trivial length/lexical baseline does not explain the judge label's
structure at chance-level AUROC -- no such computation existed anywhere
in this project at the time, and a fresh review correctly flagged this
as an unsupported claim. `code/32_surface_baseline_vs_judge_label.py`
was written to actually compute it (the result, not chance-level but not
meaningfully different from the same baseline's performance against the
original Jaccard label, is reported directly in §4).

**Direction-validity CI misdescribed as "consistent with chance," and a
10x cosine transcription error.** A subsequent, independent review found
two separate transcription/interpretation errors in §3.4: (1) the four
direction-validity bootstrap CIs at n=11 were described as "all
consistent with chance," when in fact three of the four exclude 0.5
entirely, and an exact Mann-Whitney test at this n_pos=3, n_neg=8 split
finds them nominally significant in the anti-predictive direction (L8
FFN/Attn p=0.0242, L9 FFN p=0.0121). (2) The permutation-based
cosine-similarity check reported "-0.058/-0.051"; the underlying result
file gives -0.0058/-0.0051, a 10x transcription error. Both are
corrected in §3.4. Fixing (1) in turn exposed that the power analysis
(`code/43`) used to argue this test is uninformative was itself one-sided
(only checking whether the CI's lower bound exceeds 0.5), structurally
unable to detect power in the anti-predictive direction the data
actually fall in. Recomputed two-sided, this test turns out to be
reasonably well-powered (73-99%) at exactly the AUROCs observed, meaning
the anti-predictive p-values are not simply underpowered noise. This
motivated a genuinely new check
(`code/46_direction_validity_resplit_diagnostic.py`): redrawing the
direction-fit/validity-holdout split at 200 different random seeds
(instead of the kernel's single seed) on the identical 534-item labeled
pool, with activations extracted once and cached. Across resplits, the
held-out AUROC centers near or slightly above chance (mean 0.54-0.58,
full range 0.0-1.0), and the original single seed's anti-predictive
result sits in the extreme low tail of this distribution (0.5%-4%
percentile at the four cells) -- resolving the open question the
two-sided power analysis alone could not: the original result was an
atypical, unlucky single draw, not a stable property of these
directions.

**A category-leakage check, identified as the single highest-priority
missing experiment by an independent review, run for the first time this
round.** TruthfulQA's 38 topical categories have correct-answer rates
ranging from 0% to 10.5% among the 10 most frequent categories on this
GPT-2 pool (wider still, 0%-28.6%, across all 38 including smaller
ones), so a probe correlated with
topic alone could produce an above-chance AUROC under standard random
K-fold CV with zero genuine hallucination signal -- `code/02`'s
component-probe CV protocol never checked for this.
`code/47_category_leakage_diagnostic.py` reruns the identical last-token
FFN/Attn extraction at L8/L9 on the same 534-item pool, this time with
each item's TruthfulQA category attached, and compares standard 5-fold CV
against leave-one-category-out CV (16 of 38 categories have both classes
represented in their held-out slice; the remaining 22 are skipped,
single-class, itself a consequence of the same class imbalance disclosed
in §3.2). The result: standard CV AUROC of 0.616-0.663 collapses to
0.479-0.491 -- chance or below -- at every layer and component tested.
This is now reported prominently in §3.2 and the abstract rather than
only here, given its bearing on every passive-probe number this paper
reports. A follow-up Kaggle run
(`kaggle_kernels/paper1-category-leakage-cross-arch/`,
`results/category_leakage_cross_arch_results.json`) extended this same
check to Pythia-410M and Qwen2.5-0.5B at each architecture's own peak
FFN/Attn layers, over the full 817-item TruthfulQA split: the result is
heterogeneous rather than a uniform replication of GPT-2's collapse --
Pythia's component probe is essentially unaffected (L11 FFN
0.618->0.617; L4 Attn 0.612->0.602), while Qwen0.5B partially collapses
at both components (0.566->0.485 FFN, 0.563->0.526 Attn, an untested
numerical difference between them). See §3.2
for the full result and its remaining disclosed scope (the other five
converging methods have not yet been checked under leave-one-category-out
CV for any architecture).

## References

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
