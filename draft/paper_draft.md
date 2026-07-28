# FFN Over-Retrieval Does Not Cleanly Extend to Closed-Book Confabulation: A Null Result on Component Specificity, Causal Control, and Template Invariance

**Lakshmi Chakradhar Vijayarao**
Independent Researcher
`lakshmichakradhar.v@gmail.com`

## Abstract

ReDeEP (Sun et al., ICLR 2025, arXiv 2410.11414) established a "knowledge FFN" mechanism: feed-forward
(FFN) sublayers can override retrieved context during hallucination in
retrieval-augmented generation. We test whether the same asymmetry
holds in pure closed-book generation, where there is no context to
override, across three architectures spanning a ~4.3x parameter range:
GPT-2 117M, Pythia-410M, and Qwen2.5-0.5B-Instruct.

FFN beats Attention as a hallucination-detection signal in a numerical
majority of layers on all three architectures (66.7\%, 66.7\%, 58.3\%),
but every per-architecture binomial test is non-significant
(p=0.39/0.15/0.54); pooling across all 60 layers is invalid, since
adjacent layers within an architecture are strongly autocorrelated, not
independent trials. On peak AUROC, the single best-discriminating
component is Attention on GPT-2 (L3=0.6165 vs.\ FFN L8=0.6053) and FFN
on Pythia (L11=0.6181 vs.\ Attn L4=0.6115), but on both architectures
this margin sits within the peak's own cross-validation standard
deviation ($\approx$0.03-0.06) -- not reliably distinguishable from a
tie.

Qwen2.5-0.5B, the largest model tested (~494M parameters), was
initially queried with a bare `Q: ... A:` template rather than its chat
template -- an uncontrolled out-of-distribution confound for this
instruction-tuned model. Under the bare template it showed a narrow
FFN edge (58.3\% layer-majority, AUROC $0.5657$ vs.\ $0.5625$); rerun
with its proper chat template
(`code/02_cross_arch_component_probe.py qwen05chat`,
`tokenizer.apply_chat_template`), this reverses -- FFN's layer-majority
drops to a minority (11/24=45.8\%) and Attention becomes the clear
peak-AUROC leader (L4$=0.5988\pm0.0438$ vs.\ FFN L4$=0.5704\pm0.0186$).
We report both runs rather than substitute one for the other: the
reversal is itself evidence that template choice, not scale, drove
Qwen's outlier behavior, and it revises the architecture-level
FFN-majority claim from 3/3 to 2/3 (GPT-2, Pythia).

A margin this small (0.605 vs.\ 0.576 on GPT-2) is close enough to a
question-difficulty confound to warrant a direct test.
Difficulty-matched controls (entropy-only, $n_{\text{matched}}=492/534$,
and a stronger 6-feature composite, $n_{\text{matched}}=462/534$) show
the FFN/Attention signal survives essentially unchanged on GPT-2
(entropy-match FFN $0.6085\pm0.0442$, Attn $0.6102\pm0.0285$;
composite-match FFN $0.6255\pm0.0510$, Attn $0.6253\pm0.0926$;
permutation $p=0.0020$ throughout) and replicates on Pythia (FFN
$0.6327\pm0.0513$, Attn $0.6093\pm0.0221$, both $p=0.0020$), but not on
Qwen2.5-0.5B ($p=0.230$/$0.206$) -- a genuine architecture split, not a
uniform confirmation.

We then test a harder question: does patching the FFN sublayer
specifically -- not the whole residual stream, and not just correlating
with a frozen probe -- change what a model actually generates? A
construct-validity check finds 42/81 (51.9\%, Wilson 95\% CI
[41.1\%,62.4\%]) of baseline "hallucinated" completions are degenerate
repetition loops rather than confident false claims -- confirmed
stable at the dataset's maximum $n=228$ (121/228, 53.1\%, CI
[46.6\%,59.4\%]) -- meaning the flip-to-correct metric partly measures
repetition-breaking, not semantic correction. Restricting to the 107
genuinely non-degenerate prompts at $n=228$, the causal null holds at
every tested layer/strength (McNemar $p=1.000/1.000/1.000/0.503$), and
a direct FFN-vs-Attention patch comparison shows no distinguishable
difference ($p=1.000$ throughout) -- consistent with an independent
finding (arXiv 2604.13068) that residual-stream steering fails to
correct hallucinated answers at GPT-2 scale.

A stronger causal test, ROME-style tracing (Meng et al.\ 2022) on the
45 GPT-2 examples where a question-span corruption measurably degrades
a forced-choice score, initially found a specific,
family-wise-corrected restoration effect for Attention -- not FFN -- at
layer 9 ($p=0.0026$), the same layer this paper's independent
dense-probe, sparse-probe, and steering methods already identify as
GPT-2's peak. Raising $n_{\text{valid}}$ to the dataset's maximum (67)
and pre-registering the stricter joint correction as primary, this
effect weakens below significance ($p=0.012$); instead, an
anti-specific FFN effect -- a mismatched example's activation restoring
discrimination better than the example's own -- clears the joint
threshold ($p=0.00086$). More data did not vindicate the one
borderline FFN-favoring signal this paper produced; it reversed
direction instead, reinforcing rather than complicating the overall
causal null. We flag this reversal itself as diagnostic, not just its
outcome: a single borderline effect that flips sign between a low-$n$
and a higher-$n$, more-strictly-corrected pass is at least as consistent
with both passes being underpowered draws from a genuinely null
underlying effect as it is with the second pass having found the
"true" direction. We do not have a third, independent replication at
even higher $n$ to distinguish these, and we do not claim the
higher-$n$ anti-specific-FFN result is itself a stable finding rather
than one more stopping point in a still-noisy series. Extending ROME
tracing to Pythia-410M and
Qwen2.5-0.5B-Instruct shows no surviving effect at either -- a clean
null contrasting with GPT-2's own narrow, per-family-only effect.

Two further checks close remaining explanations for these nulls. An
adversarially-trained probe (a gradient-reversal term against a
difficulty/entropy proxy, confirmed to actively discard that
information: held-out $R^2=-0.71$ FFN / $-1.44$ Attn, worse than
predicting the mean) still classifies hallucination above chance for
both components (AUROC $0.5944\pm0.0624$ FFN, $0.6134\pm0.0542$ Attn,
both $p=0.0050$) -- the signal is not question-difficulty leaking
through the representation. And thresholding this paper's one
passively-significant SAE feature ($p=4.8\times10^{-11}$ under BH-FDR
across 24,576 simultaneous tests) as a single-feature classifier
reaches AUROC$=0.5614$, with precision flat at the 4.8\% positive-class
base rate across every tested recall level (50-95\%) -- extreme
statistical significance at this scale does not translate into any
usable routing or inference-economy signal.

Together: a numerically consistent but statistically fragile passive
FFN signal, a template confound that overturns Qwen's contribution to
it, a causal intervention that is null and robust to construct-validity,
scale, and adversarial-confound checks, and a stronger causal method
(ROME) whose only significant result, at higher power, points toward
Attention specificity rather than FFN. The ReDeEP mechanism does not
cleanly extend to closed-book confabulation at this scale. All of this
rests on a Jaccard word-overlap label whose validity we also tested
directly on GPT-2: an independent LLM judge agrees with only 52\% of
labels ($\kappa=0.04$), and specifically disagrees with 47 of the 50
completions this paper's own metric calls "correct" -- a sharply
one-sided discrepancy this paper reports rather than screens out. This
concern bears most directly on absolute AUROC magnitudes throughout the
paper. The comparative findings -- the template-driven Qwen reversal,
and the causal null -- each compare two conditions scored under the
identical label scheme, so they are more robust to uniform label noise
than any single absolute AUROC value is; this is a plausibility
argument, not an independently verified claim, since we have not tested
whether the label disagreement itself is uniform across conditions.

## 1. Introduction

Why should a reader outside this specific mechanism's lineage care about
this paper? Not because it discovers a new mechanism -- it doesn't, and
says so throughout. The reason is methodological and general: this
paper is a worked example of testing whether a causal-mechanistic claim
generalizes outside the regime it was discovered in, using a specific
discipline -- a construct-validity check on the outcome metric itself,
run before any causal claim is trusted -- that is reusable well beyond
FFN sublayers or hallucination. Concretely, before this paper's causal
patching experiment is allowed to speak to "does FFN activity cause
hallucination," it must first survive the question "is the underlying
flip-to-correct metric even measuring semantic correction, or is over
half of it repetition-breaking on degenerate completions?" (\S3.4 finds
the latter, for 51.9-53.1% of baseline "hallucinated" completions). Any
researcher running a causal intervention on an LLM and reporting a
flip-rate metric inherits this same risk, independent of which
component or behavior is being tested. That transferable discipline,
not the specific FFN finding, is this paper's most durable contribution,
and it is why we believe a careful negative result belongs in a venue
that weighs correctness over novelty.

Fluent text and factual correctness are distinct properties of language
model output. ReDeEP (Sun et al., ICLR 2025) localized one mechanism by
which they diverge in retrieval-augmented generation. FFN sublayers can
override retrieved context there -- a "Knowledge FFN" over-retrieval
pattern. Does the same FFN-associated pattern operate in pure
closed-book generation, where there is no retrieved context to
override? This is a natural extension. To our knowledge, no
ReDeEP-lineage follow-up has tested it. We test it directly on GPT-2. We
replicate the test on two further architectures, Pythia-410M and
Qwen2.5-0.5B-Instruct. We then probe it causally via targeted
FFN-sublayer activation patching. The number of methods this paper
applies to what is ultimately one question is deliberate, not
padding: a passive layer-majority signal this small (a few points of
AUROC) admits several mundane alternative explanations -- an
uncontrolled prompt-template confound, a question-difficulty
confound, a construct-invalid outcome metric, an underpowered causal
test -- and each additional method exists to rule out exactly one of
these before the remaining result is taken at face value. That several
of them ruled the effect out, rather than confirming it, is the
finding; the convergent-methods design is what makes that finding
trustworthy rather than the product of a single fragile test.

The result, reported honestly rather than rounded up, is more equivocal
than a first pass suggested. Three findings drive this: a numerical FFN
majority whose per-architecture and pooled significance tests are both
compromised (the former by a one-/two-sided mislabeling, the latter by
layer autocorrelation invalidating the independence assumption -- see
§3.2-3.3); a single best-discriminating component that is actually
Attention on one of three architectures, by the paper's own primary
metric; and a causal intervention that shows zero measurable
FFN-specificity. We report this as a genuine, if modest, contribution:
what a careful test of this extension actually shows, including where
it falls short of the original hypothesis.

**Contributions, honestly scoped:**
1. Localization of hallucination signal to layers 8-9 (~66.7-75% relative
   depth on GPT-2's 12 layers) via 6 converging methods -- triangulating rigor, not a
   qualitatively new discovery (mid-layer hallucination localization is
   established prior art, engaged directly in related work). A seventh
   method (DLA magnitude) is addressed separately (\S3.1): it actually
   peaks at L10/L11, not L9, and is excluded from this count.
2. FFN-vs-Attention component decomposition testing whether ReDeEP's
   RAG-scoped FFN mechanism extends to closed-book (no-retrieval) QA. We
   find a numerical FFN majority on 3/3 architectures in the
   bare-template first pass (too few architectures for a formal test).
   Within-architecture binomial tests are two-sided non-significant
   (p=0.39/0.15/0.54). The layer-pooled test, even after correcting a
   one-/two-sided mislabeling (corrected one-sided pooled p=0.026), is
   not a valid inferential instrument: adjacent layers are not
   independent trials. We report this as suggestive at the architecture
   level and retract any claim, positive or negative, resting on the
   pooled layer-level p-value.
3. Cross-architecture data (GPT-2, Pythia-410M, Qwen2.5-0.5B) shows the
   FFN-majority pattern is directionally consistent but statistically
   fragile. We identified an instruction-tuning/template confound on the
   weakest data point, Qwen0.5B, then closed it by rerunning Qwen0.5B
   with its proper chat template. The corrected run flips both Qwen0.5B
   metrics to Attention-favoring -- peak AUROC and layer-count majority
   alike. This changes the honest 3/3-architecture count to 2/3 under
   the architecturally-correct protocol. This is not a clean scale
   story, and not even a clean 3-architecture replication once queried
   properly.
4. A targeted FFN-sublayer causal patching experiment shows a real,
   direction-consistent but non-significant effect (found vs. random
   direction). A direct FFN-vs-Attention-patch comparison shows **zero
   measurable specificity** (McNemar p=1.000 throughout). This is honest
   evidence against, not for, an FFN-specific causal mechanism at this
   sample size.

## 2. Related Work

**Parametric factual knowledge in FFN sublayers.** This section grounds
each of the paper's claims in the correct prior literature, before
making any hallucination-specific claim. The correct grounding for "FFN
sublayers are where closed-book parametric recall lives" is the
knowledge-editing literature. ROME (Meng et al. 2022) and MEMIT (Meng et
al. 2023) causally locate and edit specific facts by writing to
mid-layer FFN weight matrices. Knowledge Neurons (Dai et al. 2022)
independently identifies FFN-resident neurons whose activation
correlates with specific factual recall. Our finding is a much weaker,
aggregate, correlational claim than theirs, and we state this
explicitly. ROME, MEMIT, and Knowledge Neurons demonstrate FFN
sublayers are *causally editable* loci of individual facts. We find
only that FFN sublayer activity *as a group* is a
modestly-better-than-chance hallucination signal *on average* across a
dataset. These are not the same strength of claim. We do not conflate
them.

**RAG-scoped FFN over-retrieval.** ReDeEP (Sun et al., ICLR 2025)
established that FFN sublayers can override retrieved context during
RAG hallucination -- "Knowledge FFN" over-retrieval. Follow-up work
(ParamMute, NeurIPS 2025; SEReDeEP, arXiv 2505.07528) refines this mechanism without
leaving the RAG regime. Both rely on benchmarks that supply retrieved
context by construction. Two further ReDeEP-lineage works
remain in this same RAG-scoped regime. FACTUM (arXiv
2601.05866) reframes RAG citation hallucination as a scale-dependent
coordination failure between Attention ("reading") and FFN
("recalling") pathways. It introduces pathway-alignment scores as a
direct extension of ReDeEP's decoupling framework. RAGLens (Xiong et
al., "Toward Faithful Retrieval-Augmented Generation with Sparse
Autoencoders," arXiv 2512.08892) uses SAEs on internal activations to
detect and explain RAG unfaithfulness at the token level. Both, like
ParamMute and SEReDeEP, are confirmed RAG-only. Neither tests the
closed-book question this paper addresses. They sharpen the "how does
this work within RAG" question rather than the "does this generalize
outside RAG" one. Neither undermines this paper's closed-book novelty
claim, but both belong in this lineage list.

Whether the same FFN-associated pattern operates when there is no
context to override at all -- pure closed-book generation -- has, to
our knowledge, not been tested. This is the gap we test, with a result
that is more equivocal than a first pass suggested (§3.2-3.3).

**Detection without correction.** A recent independent finding (arXiv
2604.13068) reports two results. Output-confidence baselines beat
activation probes above ~410M parameters. Residual-stream steering
flips 0/7 tested models' generated answers toward correct, across 42
configurations on GPT-2-scale models.

We compare our probe against a matched surface baseline to measure the
same tradeoff on our own data (0.605 vs. 0.576, a 0.029 margin). The
0.576 baseline is a 6-feature surface classifier over mean and max
per-question entropy, logit variance, confidence gap, attention
entropy, and activation norm, reproduced in-project at
`code/05_run_surface_baseline.py` (MLP 5-fold CV AUROC
$=0.5756\pm0.0684$, rather than merely imported as a number). This fold
mean $\pm$ fold std is taken over 5 folds; it is not a standard error
and we do not convert it to a CI here, for the same reason \S3.7 gives
for replacing an equivalent fold-std-based z-test with a
label-permutation test -- 5 folds is too few to license a
normal-approximation interval. This margin is directionally consistent
with a weak-signal small-scale regime, but must be read alongside that
paper's stronger negative framing.

Our new FFN-sublayer causal-patching experiment (§3.4) directly tests
whether a component-targeted intervention does any better than their
generic residual-stream result. It does not (McNemar p=1.000,
FFN-patch vs. Attention-patch). This is an independent replication of
their null finding at the component level, not a refutation of it.

**Black-box, non-mechanistic detection.** This paper's entire probing
apparatus is orthogonal to, and should not be read as competing with,
the dominant hallucination-detection paradigm at the scale that
matters in practice: black-box methods that require no access to
hidden states at all. Semantic entropy (Kuhn et al., 2023,
"Semantic Uncertainty") clusters sampled generations by meaning and
flags high-entropy clusters as likely hallucinations; SelfCheckGPT
(Manakul et al., 2023) checks a single generation for consistency
against multiple stochastic resamples of the same prompt. Both operate
purely on output text, generalize across model families and closed
API-only models where hidden states are unavailable, and are the
actual production-relevant baseline this paper's mechanistic probes
would need to beat, not merely a surface-feature classifier over
internal activations. We make no such comparison here -- doing so
would require running these baselines on the same TruthfulQA items
under the same label scheme, which is outside this paper's scope -- and
we flag its absence explicitly rather than let the paper's internal
surface-baseline comparison (previous paragraph) imply completeness
against the wider detection literature it does not test against.

**Sparsity and steering method provenance.** The 100/768-dimension sparse probe result (§3.1) is
correctly attributed as follows, distinct from a superficially similar
paper it should not be confused with: "A Single Direction of Truth"
(arXiv 2507.23221) does not make the dense-vs-SAE-probe comparison at
all. It uses a
dense linear residual probe and localizes its signal to sparse
late-layer MLP activity via gradient$\times$activation attribution -- a
relevant parallel to this paper's own FFN/sparsity findings, but not
evidence for the specific dense-vs-SAE-probe claim. Kantamneni et al.,
"Are Sparse Autoencoders Useful? A Case Study in Sparse Probing" (arXiv
2502.16681), directly supports the claim instead: dense L1 probes often
match or beat trained SAE probes. That paper finds dense linear probes
perform near-perfectly, including out-of-distribution, while
comparably-sized SAE-based probes underperform. The 100/768-dimension
sparse probe result (§3.1) is consistent with this broader pattern, not
a novel sparsity discovery. The difference-of-means steering method
used throughout -- both the original whole-residual steering and the new
FFN-targeted patching -- descends directly from Inference-Time
Intervention (Li et al. 2023, arXiv 2306.03341).

**A recurring meta-pattern this paper's own confound-hunting fits.**
This paper's own confound-hunting is not an isolated instance -- the
chat-template rerun that reverses Qwen0.5B's component comparison
(§3.3), and the difficulty-matched control that turns out to test a
confound weaker than assumed (§3.7), both fit a broader pattern also
observed in three concurrently-submitted anonymous manuscripts
(citations withheld for double-blind review; summarized here
without attribution). A geometric certificate for hallucination
detectability finds its near-saturated forced-choice AUROCs are
substantially explained by answer-length surface features rather than
truthfulness: a trivial 6-feature classifier reaches AUROC 0.977 on the
identical task. An agentic-failure taxonomy finds a full-vector
hidden-state "early-warning" signal collapses to chance on one model
once step-1 difficulty is controlled for, and survives only weakly on
the other. A leakage-taxonomy audit finds that a carefully-designed
synthetic severity estimate required jointly resolving a
training-budget confound, a random-seed confound, and a miscalibrated
difficulty target before it stabilized.

Read together, the four studies support a single claim broader than any
one of them. Passive linear and geometric probes on LLM hidden states
are pervasively vulnerable to superficial confounds -- prompt template,
answer length, question difficulty, evaluation-protocol artifacts.
These confounds are easy to overlook. They consistently shrink or
eliminate the apparent signal once controlled for, across independent
model families, tasks, and probing methodologies.

## 3. Methods and Results

### 3.1 Layer localization (GPT-2, 6 converging methods, plus a corrected 7th)

**Reproducibility.** All seven numbers below are
reproducible directly from this repo, without external
dependencies on the unshipped mech-int sibling project where they were
originally computed. The small (<10KB each) per-method result summaries are
vendored at `results/vendored_mech_int/` (copied from mech-int's own
`results/logs/*.npy`, verified byte-identical via md5 where an
in-project copy already existed). `code/00_verify_vendored_mechint_numbers.py`
loads them and reprints every number below for direct verification. One
thing is not vendored: the raw per-sample hidden-state activations
these summaries were originally computed from (too large, ~2.9GB).
Recomputing the summaries from scratch, rather than verifying the
already-computed values, still requires the mech-int project itself.

Six methods converge on layers 8-9. Dense probe peak: L9 (0.5827).
Sparse L1 probe peak: L9 (100/768 active dims, 87% sparse, CV AUROC
0.589 -- not the inflated in-sample 0.874). Token-position probe peak:
L8 last-token (0.6036). Component probe: FFN peak L8 (0.6053) vs Attn
peak L3 (0.6165). Steering: peak improvement at L9. Gold-token logit
lens: divergence at L8. A seventh method, DLA magnitude, is addressed
separately below and does not join this convergence once corrected.

**Two of these six methods are weaker or more
selectively-reported than "converges on L8-9" implies.** The steering method's "peak improvement" at L9 is $0.0015$
AUROC points ($0.5759\to0.5774$), the argmax over a 13-layer
$\times$ 4-alpha (52-cell) grid
(`results/vendored_mech_int/steering_layer_sweep.npy`) -- against
per-layer cross-validation standard deviations of $0.03$-$0.08$
documented throughout this paper, this is not distinguishable from
argmax-over-noise. Separately, the logit-lens analysis
(`results/vendored_mech_int/logit_lens_results.npy`) computes two
divergence-layer estimates from the same run: the plain
correct-vs-hallucinated divergence peaks at layer 1, not layer 8; only
the gold-token variant peaks at L8, and that is the one reported above.
We report both numbers here rather than silently prefer the
convergent one: of the six methods, four (dense probe, sparse probe,
token-position probe, component probe) are real, comparatively strong
signals; the remaining two (steering, logit lens) are, respectively, a
noise-level grid-search artifact and a metric choice that agrees with
the other four only under one of two available scorings for that
method.

**DLA magnitude does not support an L9 peak.** L9 does not have the
largest absolute DLA in the underlying data; L10 (+0.90) and L11 (+0.71) are
actually larger. L8-9 is supported by the convergence of the other 6
methods above, not by DLA magnitude. The L8 FFN over-retrieval DLA
figure reported in §3.2 below is an in-sample, non-cross-validated mean
difference. It should be read as suggestive, not as confirmed evidence
of a mechanism, pending a held-out replication.

### 3.2 FFN vs. Attention component decomposition (GPT-2)

On GPT-2, FFN wins 8/12 layers (two-sided binomial p=0.39; one-sided
p=0.19; neither significant). Peak FFN layer is L8 (AUROC 0.6053); peak Attn layer
is L3 (AUROC 0.6165). **The single best-discriminating component on
GPT-2 is Attention, not FFN** -- a tension with the "FFN dominates"
framing that must be stated plainly. At L8, FFN direct logit
attribution is higher for hallucinated samples (5.08) than correct
samples (4.85). This is in-sample and not cross-validated: a suggestive
but unconfirmed "over-retrieval" signature.

### 3.3 Cross-architecture data (GPT-2, Pythia-410M, Qwen2.5-0.5B-Instruct)

[Full version: `draft/cross_architecture_section.md`, real Kaggle data,
N=605 Pythia / N=513 Qwen0.5B.] FFN
wins a numerical majority of layers on all three architectures (66.7%,
66.7%, 58.3%). The per-architecture p-values (GPT-2 0.39, Pythia 0.15, Qwen
0.54) are two-sided. The corresponding one-sided values are
0.19, 0.076, and 0.27 -- still individually non-significant.

Pooled across all 60 layers, 38/60 FFN wins gives a one-sided p=0.026,
nominally significant. We do not report this as evidence for FFN
dominance. The 60 "trials" are layers within only 3 architectures,
strongly autocorrelated within each architecture. This violates the
independence assumption the binomial test requires. The only cleanly
poolable count is at the architecture level: FFN numerical majority on
3/3. This is directionally consistent but far too small an $n$ for any
formal test.

On the paper's primary metric, empirical AUROC, Attention is the single
best-discriminating component on **one** of three architectures
(GPT-2), not two. Pythia's peak AUROC favors FFN (L11=0.6181 vs. Attn
L4=0.6115). Qwen0.5B's peak AUROC also favors FFN (L8=0.5657 vs. Attn
L17=0.5625). Qwen's \emph{Fisher-separability} result (Attn J=1.251 vs.
FFN J=1.179) is a different metric from AUROC and is not used here to
determine which component "wins" on this architecture.

**The same "within CV
SD, not distinguishable from a tie" standard applies to Pythia's margin
as well.** Pythia's FFN-vs-Attn margin
(0.6181$-$0.6115$=$0.0066) is also well within its own per-layer
cross-validation SD (FFN $0.0442$, Attn $0.0345$) -- less than a fifth
of either SD. GPT-2's own per-layer CV SD confirms the same pattern
directly, not by analogy (recovered from the parent mech-int project's
saved fold-level output, `data/processed/component_results.npy`: FFN L8
SD$=0.0557$, Attn L3 SD$=0.0427$). Its margin (0.6165$-$0.6053=0.011) is
well under a quarter of either component's own SD. **The honest,
uniform statement is: the peak-component question (FFN vs. Attention)
is within measurement noise on all three architectures tested, not just
Qwen.** There is no architecture in this study where one component's
peak AUROC advantage over the other is clearly resolved at this sample
size.

Qwen2.5-0.5B (~494M parameters -- the **largest** of the three models,
not the smallest) shows the weakest and most equivocal signal. Peak
AUROC is only 0.5657, barely above its own Fisher-geometric peak
favoring Attention at the same shared layer L12 (J=1.251 vs. FFN's
1.179) -- a real reversal on the geometric metric, even though FFN
edges out Attention on empirical AUROC at its own peak layer L8. We
flag an **open, uncontrolled confound**: Qwen2.5-0.5B-Instruct was
queried with a bare `Q: ... A:` template rather than its chat template.
This is genuinely out-of-distribution usage for an instruction-tuned
model, and a more parsimonious explanation for its weak/reversed
results than any story about scale.

**Closing the confound: we reran
Qwen2.5-0.5B-Instruct with its proper chat template**, rather than
leaving this as an open caveat
(`code/02_cross_arch_component_probe.py qwen05chat`, identical
TruthfulQA prompts and labeling threshold; only the prompt construction
changed, from `f"Q: {q}\nA:"` to
`tokenizer.apply_chat_template([{"role": "user", "content": q}],
add_generation_prompt=True)`; full per-layer output saved at
`results/cross_arch_component_probe_qwen05chat.json`). The
properly-templated run keeps fewer completions past the labeling
threshold ($n=433$ vs.\ 513) -- itself a symptom of how differently the
model behaves off- vs.\ on-template.

This rerun **reverses the peak-component result**. Attention becomes
the clear peak (L4$=0.5988\pm0.0438$) over FFN (L4$=0.5704\pm0.0186$,
same layer), where the bare-template run had FFN (barely) ahead. The
layer-count result reverses too: FFN now wins only 11/24 layers (45.8%,
a minority) versus 58.3% before. Both reversals point the same
direction. The bare-template run's FFN-favoring numbers were an
artifact of the OOD confound, not a weaker version of the same
underlying signal.

The margin (0.5988$-$0.5704=0.0284) is still smaller than Attention's
own CV SD (0.0438). So "Attention now clearly wins" would overclaim, in
the same way "FFN wins 3/3" originally did. The honest statement is
narrower: the properly-queried result is Attention-leaning where the
improperly-queried one was (barely) FFN-leaning, both within noise. The
direction of the flip is the finding, not a newly-resolved
peak-component winner.

Depth fraction of the peak FFN layer under the corrected protocol is
16.7% (L4/24), not 33.3% (L8/24) as under the bare template -- one more
respect in which the original Qwen0.5B numbers described an artifact of
prompting, not a property of the model. Depth fraction of the peak FFN
layer is not universal across architectures (66.7% GPT-2, 45.8% Pythia,
33.3% Qwen0.5B bare-template / 16.7% chat-template), echoing,
independently, GEOM-PROOF's own "depth fraction is not universal"
finding for whole hidden states.

*(A Fisher-geometric overlay, reusing GEOM-PROOF's method, was also
computed as an illustrative aside. It matches the empirical peak
exactly on GPT-2 and within one layer on Pythia. But its per-layer bound
anti-correlates with the real AUROC curve on GPT-2 (r=-0.178). With only
12 layers to argmax over, this is closer to coincidence than
corroboration. We do not treat it as independent validation of anything
in this paper and do not discuss it further.)*

\begin{figure}[h]
\centering
\includegraphics[width=0.75\textwidth]{figures/ffn-attn-comparison.pdf}
\caption{Peak AUROC for FFN vs.\ Attention across every tested condition, with error bars showing each peak's own cross-validation standard deviation (as reported in \S3.2-\S3.3). The margin between components is within one CV~SD of overlap in three of four conditions, and the one architecture that initially showed a clearer FFN edge (Qwen0.5B, bare template) reverses to favor Attention once queried with its proper chat template -- visually, no condition shows a peak-AUROC gap that survives its own measurement noise.}
\label{fig:ffn-attn-comparison}
\end{figure}

### 3.4 Causal verification: FFN-sublayer patching (GPT-2)

[Real data: `results/ffn_causal_patch_results.json`.] Difference-of-means
"truthfulness direction" computed on FFN sublayer output (train split
only), injected additively into the FFN sublayer during generation.
Tested at L8/L9, alpha in {20, 40}, against a random-direction control and
an attention-sublayer-patch control, on all 81 hallucinated test-split
prompts.

**Construct-validity check on this test set: what fraction of these 81 "hallucinated" baseline completions
are confident false claims versus degenerate repetition loops?** We
scanned each baseline completion for a repeated 4-8 word phrase
occurring 3 or more times (e.g. "I think it's the people who make the
best burgers. I think it's the people who make the best burgers...").
**42 of 81 (51.9\%, Wilson 95\% CI [41.1\%, 62.4\%]) are degenerate
repetition loops by this criterion, not confident false statements of
fact.**

This bears directly on how the causal-patching results below should be
read. For roughly half this test set, a large additive intervention
"flipping" the output to a reference-matching string is at least as
consistent with "breaking a repetition loop" as with "correcting a
hallucinated claim." The random-direction control's 7.4-14.8\% flip
rate is itself plausibly dominated by repetition-breaking rather than
semantic correction. This $n=81$ test cannot separate these two effects
post-hoc on its own -- restricting to only the genuinely confident,
non-repetitive false completions, at adequate power, is the direct
follow-up this finding motivates, and is run below at the dataset's
maximum $n=228$ (see the paragraph below
restricting to the 107 non-degenerate prompts).

| Layer | Alpha | FFN-found flip rate | FFN-random flip rate | Attn-found flip rate | McNemar p (found vs. random) | McNemar p (FFN-found vs. Attn-found) |
|---|---|---|---|---|---|---|
| L8 | 20 | 0.148 | 0.086 | 0.136 | 0.267 | 1.000 |
| L8 | 40 | 0.148 | 0.136 | 0.148 | 1.000 | 1.000 |
| L9 | 20 | 0.136 | 0.074 | 0.148 | 0.302 | 1.000 |
| L9 | 40 | 0.185 | 0.148 | 0.173 | 0.648 | 1.000 |

The found-direction FFN patch beats the random-direction control in
flip-to-correct rate, in all four tested configurations. But the
effect does not reach significance at n=81 (McNemar p ranges 0.27-1.0).
**Power:** the number of discordant
pairs (prompts where found and random disagree) in these four
comparisons is 13, 19, 15, and 19 respectively. At these
discordant-pair counts, an exact McNemar test requires roughly 75-85%
of the discordant pairs to favor one direction before reaching
$p<0.05$ (e.g., 11/13, 15/19, 12/15, 15/19). Our observed splits (9/13,
10/19, 10/15, 11/19) fall well short of that bar. This test is
underpowered to detect anything short of a large, one-sided effect at
this sample size. **A null result here should be read as "no effect
detectable at this power," not as evidence against an FFN-specific
causal mechanism.**

More importantly: **a direct FFN-found-vs-Attention-found comparison
gives McNemar p=1.000 in every single configuration.** The intervention
provides no measurable evidence of FFN-specific causal control at this
sample size. Patching attention produces indistinguishable flip rates
from patching FFN. We report this plainly as a negative result on
FFN-specificity given the data collected. This directly engages arXiv
2604.13068's finding that residual-stream steering flips 0/7 tested
models' generated answers toward correct: a component-targeted
intervention does not do meaningfully better.

The random-direction control itself flips 7.4-14.8% of prompts to
"correct" -- a substantial noise floor that the found-direction effect
sits close to. Alpha=40 is within a regime the parent project's own
residual-stream steering experiment showed can invert probe AUROC below
chance -- a representation-destroying rather than representation-steering
regime. This may explain why found and random directions become
indistinguishable at L8/alpha=40 specifically (0.148 = 0.148... vs.
random 0.136).

A non-trivial fraction of
interventions in every condition produce a degenerate/unparseable
completion rather than a clean correct-or-still-wrong answer (label
$-1$ in the per-sample data, `results/ffn_causal_patch_results.json`).
At L8/$\alpha{=}20$: 19.8% (found) vs. 21.0% (random), a 1.2-point
difference. At L9/$\alpha{=}20$: 9.9% (found) vs. 19.8% (random), a
9.9-point difference -- the opposite direction from the other three
configurations. This rises to 22.2-29.6% at $\alpha{=}40$ (differences
of 4.9 and 3.7 points). Found and random directions produce degenerate
output at similar rates in three of four configurations. This is
consistent with much of what the "flip rate" table measures being
generic generation perturbation from any large additive intervention,
not a targeted semantic correction.

**We flag explicitly, rather than average over, that L9/$\alpha{=}20$
is the one configuration where this pattern reverses.** There, the
found direction is both markedly *less* degenerate (9.9\% vs.\ 19.8\%)
and more often flip-to-correct (13.6\% vs.\ 7.4\%) than the random
control. This is exactly the joint pattern a genuine, targeted
correction would produce -- the opposite of "more noise, not more
signal." One configuration out of four, at $n=81$, is not enough to
support a positive claim. But it is also not a data point the
noise-floor story can quietly average away. The honest summary: three
of four configurations are consistent with generic perturbation noise,
and one is not. The sample size here cannot adjudicate between "the
fourth is signal" and "the fourth is a false positive at this power."
We report both readings rather than selecting the one that best fits
the null.

**Scaling up: this test is no longer $n=81$.** The
$n=81$ figure above was a hard ceiling of the original 70/30 train/test
split on GPT-2's 534-item labeled pool (268 hallucinated total). We
re-ran the identical test with a leaner 15/85 train/test split (40
correct + 40 hallucinated examples for the direction, the minimum for a
stable per-class mean, versus the remaining 228 hallucinated prompts as
test) -- the maximum $n$ this fixed dataset can supply without a
degenerate direction estimate, short of the paper's own disclosed
"$n\approx300$-$400$" aspirational target but a 2.8$\times$ increase
over $n=81$ (`code/10_ffn_causal_patch_scaled.py`,
`results/ffn_causal_patch_scaled_results.json`).

**Result: the null holds, now at adequate power, and it specifically
resolves the one configuration \S3.4's original $n=81$ pass could not
adjudicate.** All four McNemar tests are decisively non-significant
($p=0.522$, $0.868$, $1.000$, $0.659$ at L8/$\alpha{=}20$,
L8/$\alpha{=}40$, L9/$\alpha{=}20$, L9/$\alpha{=}40$ respectively), with
27-46 total discordant pairs per configuration ($b,c=22,17$; $17,19$;
$13,14$; $25,21$) -- enough pairs that a real, moderate
found-vs-random effect should have been visible if one existed. Most
notably, L9/$\alpha{=}20$ -- flagged above as the one $n=81$
configuration where found beat random on both flip-rate and
degeneracy simultaneously, explicitly left open as "signal or false
positive, this sample size cannot say" -- now shows FFN-found
\emph{trailing} FFN-random (7.5\% vs.\ 7.9\%, $p=1.000$, $b=13, c=14$
nearly perfectly balanced). At higher power, this specific configuration
resolves as noise, not signal. Baseline re-labeling confirms all 228
test prompts as genuinely hallucinated under fresh scoring (228/228),
ruling out any labeling-drift explanation for the null.

**The repetition-loop construct-validity
issue flagged for the original $n=81$ test, re-checked at
$n=228$.** We regenerated the 228 baseline completions
(identical greedy decode, no patched intervention) and re-applied the
same repetition-loop criterion (a 4-8 word phrase repeated 3+ times).
$121/228$ (53.1\%, Wilson 95\% CI [46.6\%, 59.4\%]) are degenerate
repetition loops, not confident false claims -- consistent with, not
worse than, the original $n=81$ finding (51.9\%): this is a stable
property of GPT-2's TruthfulQA failure mode, not an artifact of the
smaller original sample
(`code/14_causal_patch_scaled_degeneration_filter.py`,
`results/ffn_causal_patch_scaled_degeneration_filtered.json`).
Restricting the existing found/random labels to only the $107$
genuinely non-degenerate hallucinated prompts, the null holds
throughout, if anything more uniformly than on the full 228: McNemar
$p=1.000, 1.000, 1.000, 0.503$ at L8/$\alpha{=}20$, L8/$\alpha{=}40$,
L9/$\alpha{=}20$, L9/$\alpha{=}40$ respectively. This rules out the one
remaining construct-validity explanation for the causal null -- that it
was merely averaging genuine signal against repetition-breaking noise
across the full pool -- and this strengthens \S3.4's causal-null
finding on GPT-2 itself, independent of
and prior to the Pythia/Qwen extension below.

**Extending beyond GPT-2.** We ran the same causal
test on Pythia-410M and Qwen2.5-0.5B-Instruct
(chat-templated): difference-of-means
direction on the FFN sublayer at each architecture's own established
peak layer (Pythia L11, Qwen0.5B-chat L4, from \S3.3), found-vs-random
control, McNemar exact test, reusing the FFN/Attn component vectors
already cached from the cross-architecture probe
(\texttt{code/07\_multi\_arch\_causal\_patch.py}).

Both extensions are too underpowered to
support any claim. On Pythia (baseline hallucination rate 64.5\%), valid
pairs collapse from $n=22$ at $\alpha=10$ (found vs.\ random: 18.2\% vs.\
4.5\% flip rate, $p=0.25$) to $n=7$ at $\alpha=20$ (ranking reverses:
14.3\% vs.\ 42.9\%) to $n=0$ at $\alpha=40$ (every completion
degenerates). On Qwen0.5B-chat (baseline hallucination rate 51.1\%),
valid pairs are $n=2$, $1$, and $0$ at $\alpha=10,20,40$ -- uninformative
rather than merely underpowered, since this instruction-tuned model's
chat-style responses rarely match TruthfulQA's terse reference-answer
strings closely enough to clear the word-overlap labeling threshold even
at baseline, leaving almost no validly-labeled prompts to test.

Neither extension supports or contradicts the GPT-2 finding. Pythia's
pattern (noisy, no significant effect at low $\alpha$; degeneration at
high $\alpha$) is consistent with GPT-2's. Qwen0.5B-chat's failure is a
labeling-threshold mismatch, not evidence about the causal question
itself -- a repeat with a threshold (or LLM-judge label, as used in a
companion paper in this project) calibrated to instruction-tuned
response style is the concrete next step this null motivates.

### 3.5 ROME-style causal tracing: a stronger causal test (GPT-2)

**A stronger causal test is warranted: additive mean-shift steering
(\S3.4) is a comparatively weak causal
instrument.** We replace it here with causal tracing (Meng et al.
2022, "Locating and Editing Factual Associations in GPT"), the
field's standard method for localizing causal effect, adapted to
closed-book QA (no single clean "subject span" assumed; we corrupt the
whole question span instead of one entity). Protocol: (1) a **clean**
run of "Q: {question}\nA:" caches every layer's MLP-output and
Attn-output at the last token position, and scores a forced-choice
discrimination, $\text{logit\_diff} = \text{logit}(\text{correct
reference answer's first token}) - \text{logit}(\text{incorrect
reference answer's first token})$; (2) a **corrupted** run adds
Gaussian noise ($3\times$ GPT-2's empirical embedding std, matching
ROME's convention) to the question-span token embeddings only, and
re-scores $\text{logit\_diff}$; (3) a **restoration** sweep patches the
corrupted run's activation at each (layer, component) with the clean
run's cached activation, one at a time, and records the resulting
$\text{logit\_diff}$, normalized to a restoration score
$(\text{patched}-\text{corrupted})/(\text{clean}-\text{corrupted})$;
(4) a **specificity control** repeats step 3 but patches in a
*different, randomly-paired example's* clean activation instead of the
example's own -- if a mismatched activation restores about as well,
the effect is generic, not specific to that example's discriminative
content (`code/08_rome_style_causal_tracing.py`,
`results/rome_style_causal_tracing.json`).

**Caveat, disclosed before any result: the corruption protocol degrades
the discrimination for only 45/100 candidate examples**
($n_{\text{valid}}=45$; the other 55 show clean\_logit\_diff $\leq$
corrupted\_logit\_diff, meaning the noise did not measurably hurt the
forced-choice score for those examples, so no restoration signal is
interpretable for them). All results below are computed only on the 45
examples where corruption demonstrably worked, following the ROME
convention of excluding cases with no effect to explain.

**Result: FFN shows no specific restoration effect anywhere; Attention
does, at layers 7 and 9, and layer 9's effect converges with \S3.1's
independent finding that L9 is GPT-2's peak dense-probe, sparse-probe,
and steering layer.** Comparing own-activation restoration against the
shuffled-activation control with a paired Wilcoxon signed-rank test at
each of 12 layers $\times$ 2 components (24 tests): **jointly
Holm-Bonferroni-corrected across all 24, nothing survives** -- the
smallest p-value (Attn L9, $p=0.0026$) narrowly misses the rank-1
threshold ($0.05/24=0.00208$). Treating FFN and Attention as two
separate 12-test families (a defensible, common alternative scoping,
disclosed as a choice rather than the only correct one) changes this:
**Attn L9 survives its own family's Holm correction** (own$-$shuffled
$=+0.214$, $p=0.0026$, threshold $0.05/12=0.00417$), and **Attn L7
does not** (own$-$shuffled $=+0.250$, $p=0.0088$, threshold
$0.05/11=0.00455$) -- nominally the larger effect, but not the more
significant one at this $n$. **FFN's only within-family-surviving
result runs in the opposite direction**: MLP L9 (own$-$shuffled
$=-0.254$, $p=0.0038$, survives the $n=12$ threshold) means a
*mismatched* example's FFN activation restores the discrimination
*better* than the example's own -- an anti-specific result, the
opposite of what a causal locus would look like. We do not read this
as evidence FFN is actively harmful; more likely, at this $n$ and
layer, FFN's last-token output does not carry example-specific
discriminative content that a generic activation lacks, and the
negative sign is noise around a true value near zero, not a confirmed
directional effect -- we flag it descriptively rather than interpret it
causally. **The clean reading of what does survive multi-family
correction: Attention, not FFN, shows content-specific causal
restoration in GPT-2, and it converges on the same layer (L9) that
\S3.1's entirely independent methods (dense probe, sparse probe,
steering) already identified as special** -- itself notable, since
causal tracing and passive probing share no methodology. This is
consistent with, not contradictory to, \S3.2's finding that Attention
is GPT-2's single best-discriminating component by AUROC (L3$=0.6165$
vs.\ FFN L8$=0.6053$) -- a stronger, independent causal method now
points the same direction as the paper's passive-probe evidence,
rather than rescuing an FFN-specific causal story.

**Raising power: we raised $n_{\text{valid}}$ from 45
to the maximum the labeled pool supports (67), pre-registering the
joint (not per-family) correction as primary before rerunning --
and it does not resolve Attn L9 into significance. If anything, a
different result does.**
(`code/18_rome_style_causal_tracing_scaled.py`,
`results/rome_style_causal_tracing_scaled.json`.) At $n_{\text{valid}}=67$,
Attn L9's evidence *weakens*, not strengthens: $p=0.012$, no longer
surviving even its own family's Holm threshold, let alone the joint
one. Instead, MLP L9 -- the *anti-specific* result flagged above, where
a mismatched example's activation restores discrimination better than
the example's own -- now clears the strict joint threshold
($p=0.00086 < 0.05/24=0.00208$), still in that same anti-specific
direction (own$-$shuffled $=-0.203$). We report this plainly rather
than reframe it: more data did not vindicate the one borderline
positive result this test produced at lower $n$. It produced a
different, opposite-direction result crossing the strict threshold
instead. This is consistent with our reading of the original Attn L9
finding as likely noise around a per-family-only threshold, not a real
effect suppressed by low power -- and reinforces, rather than
complicates, this paper's overall causal-null story.

**Extending beyond GPT-2 again: this ROME-style test is no longer GPT-2-only.** We extend it to
Pythia-410M and Qwen2.5-0.5B-Instruct
(`code/09_multi_arch_rome_style_causal_tracing.py`,
`results/multi_arch_rome_style_causal_tracing.json`), reusing this
paper's own established Jaccard word-overlap labeling
(threshold $0.12$, matching \S3.4's extensions) to identify "clean"
examples fresh for each architecture, since no equivalent to GPT-2's
pre-existing generation labels exists for either model. Both
architectures have 24 layers (twice GPT-2's 12), so the per-family Holm
threshold is correspondingly stricter ($0.05/24=0.00208$ at rank 1,
vs.\ GPT-2's $0.05/12=0.00417$).

**Result: neither Pythia nor Qwen2.5-0.5B shows any layer or component
surviving correction, under either the joint (48-test) or per-family
(24-test) framing -- a clean null at both architectures, unlike GPT-2's
narrow per-family survival at Attn L9.** Pythia ($n_{\text{valid}}=39$
of 100 clean examples, found after checking 517 candidate questions):
smallest uncorrected $p=0.0645$ (Attn L20), nowhere near either
correction's threshold. Qwen2.5-0.5B-Instruct ($n_{\text{valid}}=39$ of
100, checked 635 candidates, chat-templated per \S3.4's established
finding that a bare template is an OOD confound for this model): smallest
uncorrected $p=0.0041$ (Attn L15) -- numerically close to GPT-2's
surviving Attn L9 result, but still short of Qwen's own stricter
24-test per-family threshold ($0.05/24=0.00208$). We read this as: the
one causal signal this test found anywhere (GPT-2 Attn L9, itself only
surviving under a disclosed, non-default correction scoping) does not
replicate, even at a comparably lenient per-family standard, in either
extension architecture. This strengthens, rather than weakens, the
paper's overall causal-null story -- three architectures now tested with
this stronger causal-tracing method, and the single positive signal
found anywhere is architecture-specific and borderline even on its own
terms.

### 3.6 Toward active causal control: beyond a single linear direction

Section 3.4's intervention is, by construction, a single dense linear
direction (a mean-difference vector) injected additively at a fixed
strength. At $n=81$ this design cannot detect anything short of a large
effect (\S3.4). So we cannot say whether FFN sublayers resist causal
correction in principle, or whether a single linear direction is simply
the wrong intervention class to find it with. These are different
claims. Only the second motivates a concrete next experiment.

A dense mean-difference direction averages over every latent factor
that differs between correct and hallucinated examples. Topic, length,
confidence, and (if any) a genuine truthfulness signal are all
superimposed in one vector. A hallucination-correction signal might
exist in the FFN sublayer but be carried by a small number of sparse,
near-monosemantic features, rather than the dominant directions of
variance. If so, additive steering along the dense mean-difference
direction will dilute it with everything else the direction also
captures. Sparse autoencoders (SAEs) are designed to recover exactly
this kind of structure.

**Proposed protocol.** We specify the following four-step protocol as the
natural next experiment. Step 1 (training our own SAE) was not run, for
reasons given there; Steps 2-4 were run using a substitute
for Step 1's output, with results reported afterward in "What we ran,
and what we found" below.

**Step 1 -- train an SAE on FFN sublayer output.** For layer $L \in
\{8, 9\}$, collect FFN sublayer output activations $x \in \mathbb{R}^d$
over a large unlabeled text corpus (not just the 534-prompt TruthfulQA
sample). This SAE encodes activations into a sparse, over-complete
feature space: encoder $f(x) = \text{ReLU}(W_{\text{enc}}(x -
b_{\text{pre}}) + b_{\text{enc}})$, decoder $\hat{x} = W_{\text{dec}}
f(x) + b_{\text{pre}}$, with $f(x) \in \mathbb{R}^m$ and $m \gg d$
(typically $m = 8d$ to $32d$). Train it \textbf{subject to
$\|W_{\text{dec},j}\|_2 = 1$ for every column $j$} (re-normalized after
each gradient step), minimizing $\mathcal{L} = \|x - \hat{x}\|_2^2 +
\lambda \|f(x)\|_1$.

**The unit-norm decoder
constraint is not optional.** Without it, the $\ell_1$ penalty is
ill-posed. The model can shrink $f(x)$ towards zero while inflating
$\|W_{\text{dec},j}\|$ to compensate. This reduces the penalty without
changing the reconstruction -- the standard SAE shrinkage/feature-suppression
failure mode. With decoder columns fixed to unit norm, $\ell_1$ on
$f(x)$ is a meaningful sparsity penalty on activation magnitude, not a
penalty an unconstrained decoder can evade. We also flag, rather than
specify further, that a plain ReLU-$\ell_1$ SAE is a conceptual
baseline. Top-$k$ or JumpReLU variants and dead-feature resampling are
standard 2026 practice, and a real implementation should use them.

**Step 2 -- score features for hallucination-relevance.** For each
feature $j \in \{1, \ldots, m\}$, run a two-sample test comparing
$f_j(x)$ on correct-example FFN activations against hallucinated-example
activations, using the same train-split prompts as the original
mean-difference direction. Benjamini-Hochberg FDR control operates on $p$-values, not on
the effect size (Cohen's $d_j$) directly, so ranking by $|d_j|$ alone is not sufficient for
FDR control. SAE feature
activations are zero-inflated: most features fire on a small fraction
of inputs, violating the equal-variance assumption behind a naive
$t$-test. So we use a Mann-Whitney $U$ test per feature instead (robust
to zero-inflation, and not assuming normality) to obtain $p_j$. We then
apply Benjamini-Hochberg at $q=0.05$ across all $m$ tests to control the
false discovery rate, and rank the surviving features by $|d_j|$ only
after FDR selection, not before.

**Step 3 -- intervene at the feature level, not the dense-direction
level.** Take the top-$k$ FDR-surviving features (start with $k=1$, the
single most hallucination-associated feature, to keep the intervention
interpretable). Construct the reconstruction-space edit $\Delta \hat{x}
= W_{\text{dec}} \Delta f$, where $\Delta f$ sets those $k$ coordinates
of $f(x)$ to their correct-class mean activation and leaves every other
coordinate at its original value -- a \emph{clamp}, not a free
direction. Inject $x' = x + \Delta \hat{x}$ into the FFN sublayer output
during generation.

A clamp to a specific
target is not compatible with an additional free scalar $\alpha$: $x' = x + \alpha(\Delta\hat{x})$
over- or under-shoots
the clamped target unless $\alpha=1$ by construction, so $\alpha$ is
dropped for this clamp variant. A separate, \emph{steering} variant may
instead define $\Delta f$ as the raw per-feature mean-difference
direction and retain a free $\alpha$ to scale it -- this tests whether
direction alone, not a specific target magnitude, suffices. The two
variants, clamp-to-target vs.\ scaled-direction, should be run and
reported separately, not conflated as one design.

**Step 4 -- preserve the same specificity tests.** Repeat the
found-vs-random-feature-set control (select $k$ random FDR-surviving
features instead of the top-$k$ ranked ones) and the FFN-vs-Attention
control (an SAE trained on attention sublayer output, same procedure),
exactly as in the table in \S3.4. This keeps the experiment directly
comparable to the linear-direction result, not a replacement for it.
Power the sample size to the effect this paper's own analysis implies
is needed. \S3.4 showed 13-19 discordant pairs at $n=81$ requires a
75-85\% split to reach significance. A design powered for an 8-10 point
true effect at 80\% power requires on the order of $n \approx
300$-$400$ prompts -- three to five times the current sample.

We flag this as the concrete next step this paper's own null motivates,
not as a claim that feature-level intervention already works better.
That is untested. The honest reading of \S3.4 is that a single linear
direction, at this sample size, shows no measurable FFN-specific effect
either way.

**What we ran, and what we found.** We ran Steps 2-4 of this protocol, using a
genuinely pretrained, publicly released SAE rather than training our
own (Step 1's from-scratch training, at the scale and hyperparameter
search a real implementation needs, is infeasible within this project's
compute and time budget). We substitute
\texttt{jbloom/GPT2-Small-SAEs-Reformatted}'s layer-8 SAE
(\texttt{blocks.8.hook\_resid\_pre}, $d_{\text{sae}}=24{,}576$, 32$\times$
expansion, trained on 300M tokens of OpenWebText) -- a disclosed
substitution: this SAE is trained on the \emph{residual stream} at the
same layer index this paper's other analyses target, not literally
"FFN sublayer output" as Step 1 above specifies, since no
publicly-released FFN-output SAE for GPT-2 at this layer is known to
us. \textbf{Step 2's result: 0 of 24{,}576 features survive
Benjamini-Hochberg FDR ($q=0.05$) on our own 534-example closed-book
dataset}, using the exact Mann-Whitney-then-BH procedure specified
above (\texttt{code/00\_verify\_vendored\_mechint\_numbers.py}'s
sibling script, run on Kaggle:
\texttt{kaggle\_kernels/sae-feature-causal-clamp/run\_sae\_feature\_clamp.py},
\texttt{results/sae\_feature\_clamp\_paper1.json}). No feature reaches
even passive, uncorrected significance at a level FDR-correction across
24,576 simultaneous tests can survive, so Step 3's causal clamp was
never reached -- there is no candidate feature to test. \textbf{This is
a genuine null one stage earlier than \S3.4's causal test, but it is
evidence bounded by instrument mismatch, not a clean
absence-of-mechanism result.} The substituted SAE differs from Step 1's
specification in two ways at once, not one: wrong hookpoint (residual
stream, not FFN-sublayer output) \emph{and} wrong training distribution
(300M tokens of general OpenWebText, not TruthfulQA-hallucination-specific
text). Either mismatch alone could explain 0/24,576 surviving features
without implying "no sparse FFN-specific hallucination feature exists."
We state this as "no sparse feature in
a real, externally-validated 24,576-feature dictionary is even
passively associated with hallucination status on this dataset,"
before any causal question is asked -- deliberately narrower than "FFN
lacks a sparse hallucination signature," which this test cannot
establish. We ran the identical Step 2-4
procedure on a companion paper's HaluEval Pipeline-A dataset ($n=500$,
same SAE, same layer) as a second, independent check of the pipeline
itself, not a replication of this paper's own claim: there, 331/24,576
features \emph{do} survive FDR (best feature $p=4.8\times10^{-11}$),
confirming the null above is not simply a bug in the feature-selection
code (\texttt{results/sae\_feature\_clamp\_combined.json}). Yet even
with a passively-significant feature in hand, the
causal clamp test on that dataset (found-feature vs.\ random-feature
steering, identical McNemar design to \S3.4) shows no significant
specificity at any tested strength ($p=1.000, 0.508, 1.000$ at
$\eta=10,20,40$; $n=56,49,29$) -- convergent with, not contradicting,
this paper's dense-direction causal null. Whether a passive sparse
feature exists or not, causal specificity does not follow either way,
across two independent datasets and two different feature-selection
outcomes.

### 3.7 Pre-registered protocol: dissociating difficulty from hallucination signal

This is the single most important missing
experiment for dissociating difficulty from hallucination signal (\S4). The $\approx$0.03 AUROC margin over a surface-feature
baseline (0.605 vs.\ 0.576) is small enough that a generalized
question-difficulty or effort signal remains a live, undissociated
alternative to a hallucination-specific one. **We ran this control,
using data already in hand from the parent mech-int project
(`data/processed/activations.pkl`, `features.npy`; full script:
`code/06_difficulty_matched_control.py`), rather than leaving it as a
specified-but-unrun protocol.**

**This control has two further limitations, both addressed below.** (a) Matching on a single
entropy feature does not control for the other five features feeding
the 0.576 surface baseline, so we add a second, stronger control
matched on the full 6-feature composite score. (b) The original
significance test treated 5 CV-fold standard deviations as an
independent-sample standard error, which is invalid -- replaced here
with a label-permutation test.

**Difficulty proxies (two, not one).** Control 1 measures per-question
generation difficulty via mean token-level output entropy of the
model's own greedy generation: $\delta(q) = \frac{1}{T}\sum_{t=1}^{T}
H(p_\theta(\cdot \mid q, y_{<t}))$, where $H$ is Shannon entropy over
the vocabulary. This is stored as feature 0 of the parent project's
6-dimensional surface-feature vector. Control 2 uses the out-of-fold
predicted probability of a logistic regression trained on all 6
surface features (`sklearn.model_selection.cross_val_predict`,
5-fold) -- the same composite "generic surface computation" score
underlying the 0.576 baseline itself, not just one of its six inputs.

**Equalized split.** For each proxy, we stratified all 534 prompts into
$B=10$ quantile bins. Within each bin, we subsampled correct and
hallucinated groups down to $\min(n_{c,b}, n_{h,b})$. Entropy-only
matching retains $n=492/534$ (92.1\%). Full-6-feature-composite matching
retains $n=462/534$ (86.5\%) -- lower, as expected, since the composite
score captures more of whatever correctness-correlated structure exists
in the surface features than entropy alone.

**Neither proxy was strongly correlated with correctness before
matching** (entropy: point-biserial $r=0.045$, $p=0.295$; composite:
$r=0.024$, $p=0.578$; neither significant). Both are reduced further
after matching (entropy: $r=0.009$, $p=0.841$; composite: $r=-0.004$,
$p=0.935$). **The honest reading of the "before" numbers is not that we
successfully removed a confound, but that this dataset never had a
statistically detectable difficulty confound for either proxy to
remove.** A stronger natural difficulty-correctness correlation would
be a harder and more informative test. We cannot claim this result
generalizes to a dataset where one exists.

**Result: the FFN/Attention signal survives both matches, essentially
unchanged, including the stronger full-feature control.** We refit the
identical component probe (\S3.2 methodology: mean-pooled layer output,
5-fold CV logistic regression) on each matched set at the two peak
layers. For entropy-only matching: FFN~L8 AUROC~$=0.6085\pm0.0442$,
Attn~L3 AUROC~$=0.6102\pm0.0285$ (vs.\ $0.6053\pm0.0557$ /
$0.6165\pm0.0427$ unmatched). For the stronger full-6-feature-composite
matching: FFN~L8 AUROC~$=0.6255\pm0.0510$, Attn~L3
AUROC~$=0.6253\pm0.0926$ -- if anything slightly higher, not lower,
under the stronger control.

**Significance, corrected: a label-permutation test
replaces the invalid fold-std z-test.** This test runs 500 shuffles of
the matched-set labels, refits the identical CV pipeline on each
shuffle, and computes a one-sided empirical $p$ as the fraction of
permuted AUROCs $\geq$ observed. Both components clear $p=0.0020$ under
both matching schemes -- the permutation floor at 500 shuffles, i.e.\
$0/500$ permuted AUROCs met or exceeded the observed value.

**This is evidence that FFN/Attention geometry carries
hallucination-relevant information beyond either difficulty proxy
tested**, rather than a generalized effort or surface-computation
signal. We state the caveat plainly: neither proxy represented a
strong confound in this dataset to begin with. So this is evidence of
survival under a weak test, not proof of dissociation under a strong
one. Two further honest limits apply. First, both controls here use
mech-int's pre-existing 6-feature surface set, only available for
GPT-2; the multi-arch extension in \S4 (Discussion) computes a fresh
entropy proxy instead and finds Pythia replicates this section's
survival pattern while Qwen2.5-0.5B does not -- a genuine architecture
split, not a uniform confirmation, and the full-composite match
specifically remains GPT-2-only. Second, a
genuinely difficulty-confounded dataset -- one where $\delta(q)$ or the
composite score correlates with correctness at, say, $|r|>0.2$ --
remains the decisive test this paper has not run.

**A stronger, adversarial version, for a follow-up if the matched-split
test is underpowered.** This design trains the probe with a
gradient-reversal adversarial term. Alongside the hallucination
classifier head $g_{\text{hall}}(z)$ on the shared representation $z$,
attach a second head $g_{\text{diff}}(z)$ that predicts $\delta(q)$ from
the same $z$. Optimize
$$\mathcal{L} = \mathcal{L}_{\text{hall}}(g_{\text{hall}}(z), y) -
\lambda \, \mathcal{L}_{\text{diff}}(g_{\text{diff}}(z), \delta(q)),$$
back-propagating the *negative* gradient of the difficulty-prediction
loss into $z$ -- a gradient-reversal layer. This pushes the
representation toward encoding hallucination, while actively discarding
whatever information predicts difficulty. The result is a probe whose
AUROC, if still above chance, cannot be explained by $\delta(q)$
leaking through $z$.

**We ran this adversarial protocol,
rather than leaving it specified but unrun.** A small shared encoder
($z\in\mathbb{R}^{64}$, one hidden layer) feeds a hallucination head and,
through the gradient-reversal layer above ($\lambda=1$), an
entropy-prediction head, trained jointly by 5-fold CV on the identical
FFN~L8 / Attn~L3 mean-pooled activations and entropy proxy used in the
matched-split control
(`code/17_gradient_reversal_adversarial_probe.py`,
`results/gradient_reversal_adversarial_probe.json`). The adversarial
pressure works as intended: the entropy head's held-out
$R^2$ is negative for both components ($-0.71$ FFN, $-1.44$ Attn) --
worse than predicting the mean, confirming the encoder has been pushed
to actively discard entropy-predictive information, not merely fail to
rely on it by chance. **Despite this, hallucination AUROC survives
significantly above chance for both components**: FFN
$0.5944\pm0.0624$, Attn $0.6134\pm0.0542$, both $p=0.0050$ against a
200-shuffle permutation null (0 of 200 permuted AUROCs met or exceeded
the observed value, the permutation floor at this shuffle count). This
is the strictly stronger claim the matched-split control above could
not make: the hallucination signal in FFN and Attention geometry is not
explained by $\delta(q)$ leaking through the representation, because it
survives even when the representation is adversarially trained to
discard exactly that information.

## 4. Discussion and Limitations

**Data and code availability.** All code,
cached result JSONs, and the paper source are publicly available at
`https://github.com/Lakshmi-Chakradhar-Vijayarao/ffn-mechanism-audit`.
Not every result in this paper reruns from this repository alone. The seven
\S3.1 layer-localization numbers, all `results/*.json` outputs already
computed, and every script that consumes only those cached artifacts
are self-contained. Three scripts additionally require an unshipped
sibling repository to *rerun from scratch* (as opposed to re-verifying
already-saved outputs, which needs nothing further): `code/01_ffn_causal_patch.py`
imports live code from a `mech-int` sibling project
(`MECH_INT_ROOT` environment variable, defaulting to a local path) to
regenerate labeled completions; `code/03_fisher_geometry_ffn_attn.py`
and `code/06_difficulty_matched_control.py` depend on that same
project's 2.9GB `activations.pkl`, disclosed inline as "not vendored...
a genuine external dependency," and not included with this paper;
`code/15_sae_feature_gating_utility.py` reads cached hidden states from
a second sibling project (`geom-proof`) at a hardcoded local path. We
state this plainly rather than let a reader discover it: full
end-to-end reproduction of \S3.4, \S3.7, and \S4's SAE-gating result
from raw data requires two additional private repositories not released
with this paper.

**Reproducibility map, for a reviewer checking one specific number.** Rather than requiring a search through the text for which script produced which result, the table below maps each major claim directly to its script and cached result file.

| Claim | Section | Script | Cached result |
|---|---|---|---|
| Layer localization (7 converging methods) | \S3.1 | `code/00_verify_vendored_mechint_numbers.py` | `results/*.json` (per-method) |
| FFN vs. Attention component decomposition | \S3.2 | `code/02_cross_arch_component_probe.py` | `results/cross_arch_component_probe_*.json` |
| Qwen chat-template reversal | \S3.3 | `code/02_cross_arch_component_probe.py qwen05chat` | `results/cross_arch_component_probe_qwen05chat.json` |
| Difficulty-matched control | \S3.3 | `code/06_difficulty_matched_control.py`, `code/11_multi_arch_difficulty_matched_control.py` | `results/difficulty_matched_control.json`, `results/multi_arch_difficulty_matched_control.json` |
| FFN-sublayer causal patching (main null) | \S3.4 | `code/01_ffn_causal_patch.py`, `code/10_ffn_causal_patch_scaled.py`, `code/14_causal_patch_scaled_degeneration_filter.py` | `results/ffn_causal_patch_results.json`, `results/ffn_causal_patch_scaled_results.json`, `results/ffn_causal_patch_scaled_degeneration_filtered.json` |
| ROME-style causal tracing | \S3.5 | `code/08_rome_style_causal_tracing.py`, `code/09_multi_arch_rome_style_causal_tracing.py`, `code/18_rome_style_causal_tracing_scaled.py` | `results/rome_style_causal_tracing.json`, `results/multi_arch_rome_style_causal_tracing.json`, `results/rome_style_causal_tracing_scaled.json` |
| Adversarial gradient-reversal probe | \S3.6 | `code/17_gradient_reversal_adversarial_probe.py` | `results/gradient_reversal_adversarial_probe.json` |
| SAE feature clamp | \S3.6 | `code/15_sae_feature_gating_utility.py` | `results/sae_feature_clamp_paper1.json`, `results/sae_feature_clamp_combined.json` |
| Label-validity audit (Jaccard vs. LLM judge) | \S3.7 | `code/16_llm_judge_label_noise.py` | `results/llm_judge_label_noise.json` |

- The "FFN dominates" framing is only a numerical majority, and only in
  the bare-template first pass. Per-architecture binomial tests are
  two-sided non-significant (0.39/0.15/0.54). The layer-pooled test is
  not a valid instrument regardless: 60 layers across 3 architectures
  are not 60 independent trials. This must be the headline caveat, not
  a footnote. The honest summary is "FFN wins on 3/3 architectures
  numerically under the bare-template protocol, 2/3 once Qwen0.5B is
  queried correctly, too few architectures to test formally either
  way" -- not any p-value computed by pooling layers.
- Under the properly-templated protocol, Attention is the peak-AUROC
  component on two of three architectures (GPT-2 and the corrected
  Qwen0.5B run), with only Pythia FFN-favoring. Under the original
  bare-template protocol, Attention won on only one of three (GPT-2);
  FFN won on AUROC for both Pythia and bare-template Qwen0.5B.
- Qwen0.5B is the largest model tested, not the smallest, and shows the
  weakest signal under the bare template -- an uncontrolled
  instruction-tuning/chat-template confound. Rerunning with its proper
  chat template (\S3.3) confirms the confound was real, reversing both
  of Qwen0.5B's component-comparison metrics.
- The causal patching effect is directionally real -- found beats random
  -- but non-significant and underpowered at $n=81$ (13-19 discordant
  pairs per configuration, requiring a 75-85\% split to reach $p<0.05$;
  the original splits fell well short). Re-run at $n=228$ (the maximum
  this fixed 534-item labeled pool supports), the null holds at
  adequate power throughout (McNemar $p=0.522, 0.868, 1.000, 0.659$),
  and specifically resolves the one $n=81$ configuration
  (L9/$\alpha{=}20$) this paper had flagged as possible signal -- now
  shown to be noise (\S3.4). Restricting further to the 107 prompts
  confirmed non-degenerate at $n=228$ (53.1\%, Wilson CI
  [46.6\%,59.4\%], of baseline completions are repetition loops, not
  confident false claims), the null holds if anything more uniformly
  ($p=1.000,1.000,1.000,0.503$). We report this as a genuine null at
  adequate power, not merely "no detectable FFN-specificity at low
  power."
- Two difficulty-matched controls (\S3.7) -- entropy-only
  ($n_{\text{matched}}=492/534$) and a stronger full-6-feature-composite
  match ($n_{\text{matched}}=462/534$) -- show the FFN/Attn component
  signal survives both essentially unchanged on GPT-2 (entropy-match
  FFN $0.6085\pm0.0442$ / Attn $0.6102\pm0.0285$; composite-match FFN
  $0.6255\pm0.0510$ / Attn $0.6253\pm0.0926$, vs.\ $0.6053\pm0.0557$ /
  $0.6165\pm0.0427$ unmatched, permutation $p=0.0020$ throughout). The
  honest framing is survival-under-a-weak-test, not
  proof-of-dissociation-under-a-strong-one: neither proxy correlated
  significantly with correctness before matching (entropy $r=0.045$,
  $p=0.295$; composite $r=0.024$, $p=0.578$), so this dataset never
  contained a statistically detectable difficulty confound for either
  proxy to remove. Extending the same control to Pythia-410M and
  Qwen2.5-0.5B (fresh mean-entropy computed from each model's own
  greedy-decode logits, refit at each architecture's own peak layer,
  over all 817 TruthfulQA questions;
  `code/11_multi_arch_difficulty_matched_control.py`,
  `results/multi_arch_difficulty_matched_control.json`) splits by
  architecture: Pythia replicates GPT-2's survival (FFN
  $0.6327\pm0.0513$, Attn $0.6093\pm0.0221$, both $p=0.0020$, at 89.3\%
  retention), but Qwen0.5B does not (FFN $0.5277\pm0.0434$, $p=0.230$;
  Attn $0.5282\pm0.0320$, $p=0.206$, at 88.3\% retention), consistent
  with both AUROCs sitting much closer to chance there. We read this as
  a genuine open question -- component specificity is
  architecture-dependent rather than universal (\S3.3) -- not a
  labeling-threshold artifact, since this entropy pipeline does not
  depend on chat-template response length the way word-overlap
  correctness labeling does. A stronger, adversarial version of this
  same test (\S3.7: gradient-reversal against the entropy proxy,
  confirmed to actively discard it, held-out $R^2<0$ for both
  components) shows hallucination AUROC survives significantly (FFN
  $0.5944$, Attn $0.6134$, both $p=0.0050$) even under this adversarial
  pressure -- the strictly stronger claim the matched-split control
  alone could not make.
- The causal-patching null (\S3.4) is a property of a single dense
  linear intervention, not evidence that FFN sublayers resist causal
  correction in principle. An SAE feature-level intervention at a
  different granularity (a pretrained SAE substituted for training one
  from scratch, which was infeasible) finds 0/24,576 features survive
  FDR on this paper's own 534-example dataset -- a genuine null one
  stage before any causal question is asked. The identical procedure
  run on a companion paper's separate HaluEval dataset, as a
  positive-control check that the feature-selection code itself works,
  does find 331/24,576 significant features there, but even with a
  passively-significant feature in hand, the causal clamp test shows no
  specificity at any strength ($p=1.000, 0.508, 1.000$) -- convergent
  with, not contradicting, this paper's dense-direction causal null.
  Extensions of the causal test to Pythia-410M and Qwen0.5B-chat (\S3.4)
  are real attempts, but both are severely underpowered ($n\leq22$),
  for two different and instructive reasons: Pythia degenerates
  entirely by $\alpha=40$; Qwen0.5B-chat's baseline responses rarely
  clear the word-overlap labeling threshold at all. Neither supports a
  claim for or against the GPT-2 result.
- Small-model (117M-494M), single-domain (TruthfulQA),
  Jaccard-word-overlap-labeled scope throughout -- surface-form
  divergence, not verified factual incorrectness. Template scope is
  single-template for GPT-2 and Pythia; Qwen2.5-0.5B is tested under
  both a bare template and its proper chat template specifically
  because the reversal this comparison exposed (\S3.3) was itself a
  finding, not a uniform single-template design across all three
  architectures. We quantified this
  directly on GPT-2: an independent LLM judge (Qwen2.5-3B-Instruct,
  judging against TruthfulQA's reference `best_answer`) scored a
  100-item stratified sample (50 Jaccard-correct, 50
  Jaccard-hallucinated) with only 52\% raw agreement and Cohen's
  $\kappa=0.04$ -- next to no agreement beyond chance
  (`code/16_llm_judge_label_noise.py`,
  `results/llm_judge_label_noise.json`). The disagreement is sharply
  asymmetric, not symmetric noise: the judge agrees with 49/50 (98\%)
  of Jaccard-\emph{hallucinated} calls, but with only 3/50 (6\%) of
  Jaccard-\emph{correct} calls -- it independently labels 47 of the 50
  GPT-2 completions this paper's own metric calls "correct" as
  hallucinated instead. We do not read this as proof the judge is
  right and the word-overlap label is wrong: GPT-2 is a non-instruction-tuned
  base model whose raw continuations past a bare `Q: ... A:` prompt are
  often rambling rather than answer-shaped, which may itself confuse an
  instruction-tuned judge's binary framing, independent of any actual
  hallucination. But this is a real, sharply one-sided discrepancy, not
  a reassuring near-miss, and it means this paper's "correct" label on
  GPT-2 specifically should be read as "cleared a word-overlap
  threshold," not as "an independent judge would also call this
  correct." This audit covers GPT-2 only; whether the same asymmetry
  holds for Pythia-410M or Qwen2.5-0.5B remains untested.
- No inference-economy claim. This paper localizes a hallucination
  signal and tests a causal intervention; it does not propose an
  early-exit, routing, or compute-saving mechanism. None of its AUROCs
  (0.53-0.62) are strong enough to gate anything at usable precision,
  even if one were proposed. We tested this directly on the one
  passively-significant signal this project produced -- the SAE feature
  from \S3.6's companion-dataset positive control (best feature,
  $p=4.8\times10^{-11}$ under BH-FDR across 24,576 simultaneous tests).
  Thresholding its raw activation as a single-feature classifier
  reaches AUROC$=0.5614$ -- itself weak -- with precision at every
  tested recall level (50-95\%) flat at $0.048$, exactly the
  positive-class base rate
  (`code/15_sae_feature_gating_utility.py`,
  `results/sae_feature_gating_utility.json`). Extreme statistical
  significance under simultaneous testing at $n=500$ does not translate
  into any usable gating concentration: this feature would not enrich
  a routed subset above chance at any operating point.

## 5. Conclusion

Whether ReDeEP's RAG-scoped FFN mechanism extends to closed-book
generation remains an open question after this study, not a confirmed
finding. FFN shows a directionally consistent but individually
non-significant numerical majority across three architectures, under a
bare-template first pass. A layer-pooled test is not statistically
valid here, due to within-architecture autocorrelation, regardless of
whether the one- or two-sided p-value is used. The
honestly poolable fact from that pass is that 3/3 architectures show a
numerical majority -- too small an $n$ to test formally.

Correcting for the Qwen0.5B chat-template confound (\S3.3)
drops that count to 2/3. It also flips the single best-discriminating
component to Attention, on two of three architectures (GPT-2 and
properly-templated Qwen0.5B), by the paper's primary AUROC metric, with
only Pythia FFN-favoring.

A direct causal test shows no measurable FFN-specificity (p=1.000
throughout). This extends, rather than contradicts, a recent
independent finding that activation interventions fail to causally
correct hallucinated answers at this model scale.

We report this candidly as a modest, largely null-leaning contribution.
Closed-book FFN over-retrieval is a plausible but empirically
unconfirmed extension of ReDeEP. The paper's main value is in what it
honestly rules out -- clean FFN-specific causal control, a clean scale
story -- rather than what it positively establishes.

## References

Full citations below, compiled from exactly the
bibliographic detail already verified in-text or in
`related_work/related_work_notes.md`; entries with no author list
recorded anywhere in this project are cited by title only rather than
inventing names.

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
