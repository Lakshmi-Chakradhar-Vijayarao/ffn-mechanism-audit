# FFN Over-Retrieval Does Not Cleanly Extend to Closed-Book Confabulation: A Null Result on Component Specificity, Causal Control, and Template Invariance

**Lakshmi Chakradhar Vijayarao**
Independent Researcher
`lakshmichakradhar.v@gmail.com`

## Abstract

ReDeEP (Sun et al., ICLR 2025, arXiv 2410.11414) established a "knowledge
FFN" mechanism: feed-forward (FFN) sublayers can override retrieved
context during hallucination in retrieval-augmented generation. We test
whether the same asymmetry appears in pure closed-book generation, across
three architectures spanning a ~4.3x parameter range: GPT-2 117M,
Pythia-410M, and Qwen2.5-0.5B-Instruct.

FFN beats Attention as a hallucination-detection signal in a numerical
majority of layers on all three architectures (66.7\%, 66.7\%, 58.3\%),
but every per-architecture test is non-significant, and every
peak-AUROC margin sits within that peak's own cross-validation noise --
not reliably distinguishable from a tie on any architecture tested.
Qwen2.5-0.5B (the largest model, at ~494M parameters) was initially
queried with a bare `Q: ... A:` template rather than its chat template;
correcting this reverses both of its component-comparison metrics
(peak-AUROC leader and layer-majority) and drops the architecture-level
FFN-majority count from 3/3 to 2/3 -- evidence the original result was a
template artifact, not a scale effect.

Two difficulty-matched controls (entropy-only and a stronger 6-feature
composite) and an adversarially-trained gradient-reversal probe rule out
a generic question-difficulty confound as the explanation for the
residual signal on GPT-2 and Pythia, though the same control does not
replicate on Qwen2.5-0.5B. A direct causal test -- patching the FFN
sublayer specifically during generation, at an adequately powered
$n=228$ -- shows zero measurable FFN-specificity against an
Attention-patch control (McNemar $p=1.000$ throughout), consistent with
an independent finding that residual-stream steering fails to correct
hallucinations at this scale. A stronger causal method (ROME-style
tracing) finds only one effect anywhere across three architectures that
survives correction, and it favors Attention, not FFN. A sparse-feature
(SAE) intervention finds no passively-associated feature at all, bounded
by instrument mismatch rather than a clean absence of mechanism.

All of this rests on a Jaccard word-overlap hallucination label whose
validity we audited directly, on all three architectures: an independent
LLM judge agrees with only 52-58\% of labels (Cohen's $\kappa=0.03$-$0.14$
across GPT-2, Pythia, and both Qwen0.5B variants, next to chance
throughout), consistently biased toward calling completions
hallucinated that the word-overlap heuristic called correct. Rather than
leave this as an unresolved caveat, we reran the decisive causal test
(§3.4) end to end under the validated label, at nearly double the
sample size ($n=467$): the causal null does not weaken, it is
corroborated by two signatures at once -- flip rates for FFN-found,
FFN-random, and Attn-found alike collapse to a 1.3-3.0\% floor, with no
configuration distinguishable from any other even at that floor. The
passive component-comparison picture is more mixed under the
validated label: absolute AUROCs rise substantially (to 0.66-0.75) and
FFN's numerical majority is restored on Qwen0.5B-chat, but which
component leads remains architecture-dependent either way.

The ReDeEP mechanism does not cleanly extend to closed-book confabulation
at this scale. The paper's most transferable contribution is
methodological: a construct-validity check on an outcome metric, run
before trusting a causal claim built on it, catches that 51.9-53.1\% of
baseline "hallucinated" completions are degenerate repetition loops
rather than confident false claims -- a discipline reusable well beyond
this specific mechanism.

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
1. Localization of hallucination signal to layers 8-9 on GPT-2 (6
   converging methods, triangulating rigor rather than a qualitatively
   new discovery -- mid-layer hallucination localization is established
   prior art). A seventh method (DLA magnitude) peaks at L10/L11 instead
   and is excluded from this count (§3.1).
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
   measurable FFN-vs-Attention specificity (McNemar $p=1.000$
   throughout), confirmed rather than weakened when the same test is
   rerun end to end under a validated label at nearly double the sample
   size (§3.4).

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
residual-stream result; it does not (McNemar $p=1.000$) -- an independent
replication of their null at the component level.

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

### 3.2 FFN vs. Attention component decomposition (GPT-2)

On GPT-2, FFN wins 8/12 layers (two-sided binomial $p=0.39$; one-sided
$p=0.19$; neither significant). Peak FFN layer is L8 (AUROC 0.6053); peak
Attn layer is L3 (AUROC 0.6165). **The single best-discriminating
component on GPT-2 is Attention, not FFN.** At L8, FFN direct logit
attribution is higher for hallucinated samples (5.08) than correct
samples (4.85) -- an in-sample, non-cross-validated, suggestive but
unconfirmed "over-retrieval" signature.

### 3.3 Cross-architecture data (GPT-2, Pythia-410M, Qwen2.5-0.5B-Instruct)

[Full version: `draft/cross_architecture_section.md`; real Kaggle data,
$N=605$ Pythia / $N=513$ Qwen0.5B.] FFN wins a numerical majority of
layers on all three architectures (66.7\%, 66.7\%, 58.3\%);
per-architecture two-sided p-values are 0.39/0.15/0.54, one-sided
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
`results/llm_judge_relabel_summary.json`). Absolute AUROCs rise
substantially under the validated label (Pythia $0.615\to0.735$/$0.749$
FFN/Attn peak; Qwen0.5B-bare $0.566\to0.713$/$0.699$; Qwen0.5B-chat
$0.570$/$0.599\to0.660$/$0.641$), and FFN's numerical majority is
restored on Qwen0.5B-chat specifically (11/24 layers under Jaccard,
a minority, to 18/24 under the validated label, a majority) --
the template-reversal finding above is thus itself sensitive to which
label is used. Which component peaks, however, remains
architecture-dependent under the validated label exactly as it was
before (Attention still peaks on Pythia; FFN still peaks on both Qwen0.5B
variants), so this re-check strengthens the case that a real, larger
signal than the noisy label suggested exists, without resolving the
paper's core FFN-vs-Attention specificity question either way.

\begin{figure}[h]
\centering
\includegraphics[width=0.75\textwidth]{figures/ffn-attn-comparison.pdf}
\caption{Peak AUROC for FFN vs.\ Attention across every tested condition, with error bars showing each peak's own cross-validation standard deviation. The margin between components is within one CV~SD of overlap in three of four conditions, and the one architecture that initially showed a clearer FFN edge (Qwen0.5B, bare template) reverses to favor Attention once queried with its proper chat template.}
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

**Result, at adequate power.** At the maximum supportable $n=228$ (27-46
discordant pairs per configuration -- enough that a real, moderate effect
should be visible if one existed), all four McNemar tests are decisively
non-significant ($p=0.522, 0.868, 1.000, 0.659$ at L8/$\alpha{=}20$,
L8/$\alpha{=}40$, L9/$\alpha{=}20$, L9/$\alpha{=}40$). Restricting further
to the 107 prompts confirmed non-degenerate, the null holds if anything
more uniformly ($p=1.000, 1.000, 1.000, 0.503$) -- ruling out the one
remaining construct-validity explanation for the null. **A direct
FFN-found-vs-Attention-found comparison gives McNemar $p=1.000$ in every
configuration at every sample size tested**: patching Attention produces
indistinguishable flip rates from patching FFN, a negative result on
FFN-specificity given the data collected. (The original, underpowered
$n=81$ pass showed a directionally consistent found-beats-random effect,
13-19 discordant pairs per configuration, that did not reach significance
given the 75-85\% split needed at that $n$; one configuration,
L9/$\alpha{=}20$, was flagged there as possible signal, but resolves as
noise at $n=228$, 7.5\% vs.\ 7.9\%, $p=1.000$.)

A non-trivial fraction of interventions in every condition produce a
degenerate/unparseable completion rather than a clean correct-or-wrong
answer (19.8-29.6\% across conditions), found and random directions at
similar rates in three of four configurations -- consistent with much of
the flip-rate signal being generic generation perturbation from any large
additive intervention, not targeted semantic correction.

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
with the GPT-2-only audit), and we reran the identical causal-patching
protocol -- found-direction computed from a judge-labeled train split,
patched generation, every output scored by the same judge -- on all 467
judge-hallucinated test prompts (nearly double the $n=228$ used above).
Under this validated label, flip-to-correct rates collapse to
1.3-3.0\% for FFN-found, FFN-random, and Attn-found alike at every
layer/alpha, with no configuration distinguishable from any other
(McNemar $p\geq0.29$ throughout; 11-22 discordant pairs per
configuration -- fewer than the 27-46 above, since the much lower
absolute flip rate leaves fewer prompts where found and random disagree
at all). This is not a higher-powered null in the discordant-pairs
sense; it is a null corroborated by two different signatures at once --
a floor-level absolute flip rate, and zero separation between
conditions even at that floor -- under the Jaccard-labeled result above
(which shows much higher but equally undifferentiated flip rates,
33-42\% across all three conditions). The gap between the two labels'
absolute flip rates is itself informative: a large fraction of what
Jaccard counts as "flipped to correct" is apparently satisfied by
superficial word overlap with the reference that a validated judge does
not accept as genuinely correct, consistent with this paper's own
repetition-loop and degenerate-output findings elsewhere in this
section. Full methodology and per-configuration numbers:
`kaggle_kernels/paper1-causal-patch-judge-label/`,
`results/causal_patch_judge_label_results.json`.

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
anywhere. Attention's strongest candidate (L9, own$-$shuffled $=+0.214$)
does not survive the joint Holm-Bonferroni threshold ($p=0.012$ vs.\
$0.05/24=0.00208$) -- weaker than in the paper's original, lower-powered
$n=45$ pass, where it had survived only a less conservative per-family
(not joint) scoping. Instead, MLP L9 -- an *anti-specific* result, where a
mismatched example's activation restores discrimination *better* than
the example's own -- clears the strict joint threshold
($p=0.00086$, own$-$shuffled $=-0.203$). More data did not vindicate the
one borderline FFN-favoring signal this test produced; it reversed
direction instead. We read this as consistent with the original Attn L9
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
(`kaggle_kernels/sae-feature-causal-clamp/run_sae_feature_clamp.py`), so
the causal clamp step was never reached. This is a genuine null one
stage earlier than §3.4's causal test, but bounded by instrument
mismatch, not a clean absence-of-mechanism result: either axis of
mismatch alone could explain zero surviving features without implying no
sparse FFN-specific signal exists. Running the identical procedure on a
companion paper's HaluEval dataset ($n=500$, same SAE, same layer) as a
positive-control check of the pipeline itself finds 331/24,576 features
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

## 4. Discussion and Limitations

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
| Qwen chat-template reversal | §3.3 | `code/02_cross_arch_component_probe.py qwen05chat` | `results/cross_arch_component_probe_qwen05chat.json` |
| Difficulty-matched control | §3.7 | `code/06_difficulty_matched_control.py`, `code/11_multi_arch_difficulty_matched_control.py` | `results/difficulty_matched_control.json`, `results/multi_arch_difficulty_matched_control.json` |
| FFN-sublayer causal patching | §3.4 | `code/01_ffn_causal_patch.py`, `code/10_ffn_causal_patch_scaled.py`, `code/14_causal_patch_scaled_degeneration_filter.py` | `results/ffn_causal_patch_results.json`, `results/ffn_causal_patch_scaled_results.json`, `results/ffn_causal_patch_scaled_degeneration_filtered.json` |
| ROME-style causal tracing | §3.5 | `code/08_rome_style_causal_tracing.py`, `code/09_multi_arch_rome_style_causal_tracing.py`, `code/18_rome_style_causal_tracing_scaled.py` | `results/rome_style_causal_tracing.json`, `results/multi_arch_rome_style_causal_tracing.json`, `results/rome_style_causal_tracing_scaled.json` |
| Adversarial gradient-reversal probe | §3.7 | `code/17_gradient_reversal_adversarial_probe.py` | `results/gradient_reversal_adversarial_probe.json` |
| SAE feature clamp | §3.6 | `code/15_sae_feature_gating_utility.py` | `results/sae_feature_clamp_paper1.json`, `results/sae_feature_clamp_combined.json` |
| Label-validity audit (all 3 architectures) | §4 | `code/16_llm_judge_label_noise.py`, `code/23_regenerate_completions_for_judge.py`, `code/24_llm_judge_score_all_architectures.py` | `results/llm_judge_label_noise.json`, `results/llm_judge_relabel_summary.json` |
| Causal patching under validated label | §3.4 | `kaggle_kernels/paper1-causal-patch-judge-label/` | `results/causal_patch_judge_label_results.json` |

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
are degenerate repetition loops, as "correct." A trivial length/lexical
baseline does not explain the judge label's structure (chance-level
AUROC), meaning it is not simply rewarding shorter or blander
completions. Given this, we reran the paper's decisive causal test under
the validated label rather than leave the concern unresolved (§3.4): the
causal null holds, corroborated by a floor-level flip rate and zero
separation between conditions at that floor, at nearly double the
original sample size. This paper's "correct" label should still be read
as "cleared a word-overlap threshold," not as "an independent judge
would also call this correct" -- but the paper's central causal claim no
longer depends on that distinction mattering. One residual limitation:
the same judge model both defines the found-direction's train split and
scores every generated output in §3.4's validated-label test, so that
result's validity rests on the judge's own accuracy, which we have
checked only by manual spot-reading and a chance-level surface-feature
control, not an independent second judge or human annotation.

**No inference-economy claim.** This paper localizes a signal and tests
a causal intervention; it does not propose an early-exit, routing, or
compute-saving mechanism, and none of its AUROCs (0.53-0.62) are strong
enough to gate anything at usable precision. We tested this directly on
the one passively-significant signal this project produced (the SAE
feature from §3.6's positive control, $p=4.8\times10^{-11}$):
thresholding it as a single-feature classifier reaches AUROC$=0.5614$
with precision flat at the 4.8\% base rate across every tested recall
level (`code/15_sae_feature_gating_utility.py`) -- extreme statistical
significance under simultaneous testing does not translate into any
usable gating concentration.

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
A direct causal test shows no measurable FFN-specificity ($p=1.000$
throughout), extending rather than contradicting an independent finding
that activation interventions fail to causally correct hallucinated
answers at this model scale. This null does not depend on trusting the
paper's word-overlap label: relabeling every completion with an
independent LLM judge and rerunning the causal test end to end at nearly
double the sample size leaves the null intact, corroborated by both a
floor-level flip rate and zero separation between conditions at that
floor -- the strongest single piece of evidence in this paper, precisely
because it is the one result shown to survive the paper's own most
serious methodological objection.

We report this candidly as a modest, largely null-leaning contribution.
Closed-book FFN over-retrieval is a plausible but empirically unconfirmed
extension of ReDeEP. The paper's main value is in what it honestly rules
out -- clean FFN-specific causal control, a clean scale story -- rather
than what it positively establishes.

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
