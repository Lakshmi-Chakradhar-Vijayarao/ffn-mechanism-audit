# FFN Mechanism Audit

Code, cached results, and paper source for "Uninterpretable Nulls in
Interpretability Intervention Studies: A Closed-Book Case Study and a Validity
Checklist."

The paper's thesis is methodological. It reports that a closed-book test of an
FFN-vs-Attention hallucination asymmetry (analogous to ReDeEP's RAG-scoped
mechanism) **cannot be answered on this testbed**, and gives the diagnostics
that establish why: a competence ceiling (GPT-2 answers 27/817 = 3.3% of
TruthfulQA validation items correctly under an LLM judge), degeneracy
contamination of the flip-rate outcome metric (~52-54% of nominally
hallucinated completions are repetition loops), and a held-out
direction-validity gate that cannot pass at the available number of positives
(with an exact minimum-detectable-effect table, reported as a function of both
class counts). A fold-seed sensitivity sweep on the passive side shows the
peak-versus-peak FFN-vs-Attention margins (0.003-0.011 AUROC) are an order of
magnitude smaller than the spread a single cross-validation fold seed
produces. The constructive output is a ten-item validity checklist (eight
items for causal-patching studies, two for passive probing), written to be
used independently of anything about FFNs. It also reports, in full, a leave-one-category-out diagnostic
whose interpretation changed three times under scrutiny: two permutation
controls show the collapse is specific to real topic structure (not an
estimator artifact); a topic-only ceiling calculation rules out the originally
claimed per-category-base-rate mechanism; and an estimand-matched
recomputation shows the probe's within-topic discrimination is at chance
whether or not same-topic items are in training, so the "training-set topic
overlap" residual an earlier version reported was an artifact of comparing a
pair-weighted pooled AUROC against a category-averaged one. The surviving
claim is narrower than any earlier reading — this probe's discrimination lives
in between-topic pairs, not within-topic ones.

## Start here — the four validity checks

| Check | Script | Result JSON |
|---|---|---|
| Competence ceiling | `kaggle_kernels/paper1-causal-patch-enlarged-pool/` | `.../output/causal_patch_enlarged_pool_results.json` |
| Degeneracy pre-filter + non-degenerate re-probe | `code/04_degeneration_check.py`, `code/14_...`, `code/49_nondegenerate_subset_probe.py` | `results/ffn_causal_patch_scaled_degeneration_filtered.json`, `results/nondegenerate_subset_probe.json` |
| Direction-validity gate: exact MDE/power table | `code/51_direction_validity_mde_table.py` | `results/direction_validity_mde_table.json` |
| Permutation controls + topic-only ceiling for the LOGO diagnostic | `code/48_permuted_pseudocategory_control.py` | `results/permuted_pseudocategory_control.json` |
| Estimand-matched within-topic AUROC (training-overlap residual) | `code/52_estimand_matched_within_topic_auroc.py` | `results/estimand_matched_within_topic_auroc.json` |

Two supporting diagnostics:

| Check | Script | Result JSON |
|---|---|---|
| CV fold-seed sensitivity (and the 0.6053-vs-0.643 gap decomposition) | `code/50_cv_seed_sensitivity_sweep.py` | `results/cv_seed_sensitivity_sweep.json` |
| Direction-validity 200-resplit diagnostic | `code/46_direction_validity_resplit_diagnostic.py` | `results/direction_validity_resplit_diagnostic.json` |
| Label-permutation null for that resplit diagnostic | `code/53_resplit_permutation_null.py` | `results/resplit_permutation_null.json` |
| Direction-validity gate at the available negative count (3:475, not 3:8) | `code/54_enlarged_negative_holdout_gate.py` | `results/enlarged_negative_holdout_gate.json` |

## What's included
- `draft/` — the paper source: `paper_draft.md` (markdown mirror of the LaTeX
  submission) and `cross_architecture_section.md` (the full cross-architecture
  write-up, referenced from the main draft).
- `code/` — all analysis and experiment scripts, numbered roughly in
  pipeline/chronological order. **The numbering has gaps** (e.g. `18`→`20`,
  `32`→`37`): earlier-numbered scripts were superseded by later, corrected
  versions during iterative review (documented in the paper's Appendix A,
  "Correction History"), or belong to stages whose outputs are already cached
  in `results/*.json`. `05_run_surface_baseline.py` and
  `05_surface_baseline_classifier.py` intentionally share a number — the former
  is a thin driver that imports the latter as a helper module.
- `results/*.json` — every cached result underlying every number reported in
  the paper. These are sufficient to re-verify all reported statistics without
  re-running any GPU computation.
- `kaggle_kernels/` — every Kaggle GPU run this paper depends on, one
  subdirectory per kernel (script + `kernel-metadata.json` + downloaded
  output). `KAGGLE_RUN_INSTRUCTIONS.md` is an early planning note covering only
  the original cross-architecture replication (Pythia/Qwen0.5B extraction) —
  see this directory itself for the complete, current set of runs.
- `related_work/`, `requirements.txt`.

## Superseded artifacts, retained deliberately
- `code/47_category_leakage_diagnostic.py` and
  `results/category_leakage_diagnostic.json` produce the leave-one-category-out
  numbers that an earlier draft read as per-category-base-rate leakage.
  `code/48_permuted_pseudocategory_control.py` shows that specific mechanism is
  ruled out (topic identity alone is at or below chance once cross-validated)
  while the collapse itself is real and topic-specific. Both are kept so the
  revision is auditable; see the paper's §4.10.
- `kaggle_kernels/paper1-category-leakage-cross-arch/` and
  `results/category_leakage_cross_arch_results.json` extended the same
  (invalid) diagnostic across architectures. That analysis is **no longer part
  of the paper** — beyond depending on the retracted diagnostic, it was
  confounded by three simultaneous protocol differences (last-token vs.
  mean-pooled extraction, judge vs. Jaccard labels, 5.1% vs. ~47% positive
  rate) fully aliased with architecture. The files are retained for the audit
  trail only.

## What's excluded, and why
Raw per-sample hidden-state feature caches (`results/*.npz`, tens to ~100MB
each) are excluded from this repository. These are intermediate GPU-extraction
outputs, not final results — every statistic computed from them is already
saved in the small `results/*.json` files above. Scripts that consume them
regenerate them on first run (`code/48`, `code/49`, `code/50`), or require the
earlier feature-extraction stage (`02_cross_arch_component_probe.py`), which
needs GPU access and model weights.

Four scripts (`code/01_ffn_causal_patch.py`, `code/03_fisher_geometry_ffn_attn.py`,
`code/06_difficulty_matched_control.py`, `code/15_sae_feature_gating_utility.py`)
plus the feature-cache step of `code/49_nondegenerate_subset_probe.py`
additionally depend on unshipped sibling projects (`mech-int`, `geom-proof`)
for full from-scratch reproduction — disclosed in the paper's "Data and code
availability" section.

## Verifying a reported number without GPU access
Every number in the paper traces to a small, included `results/*.json` file.
Open the relevant JSON directly rather than re-running the pipeline. The
paper's Reproducibility map lists the script→JSON mapping for every claim.
