# FFN Mechanism Audit

Code, cached results, and paper source for "FFN Over-Retrieval Does Not Cleanly
Extend to Closed-Book Confabulation."

## What's included
- `draft/` — the paper source: `paper_draft.md` (canonical, markdown mirror
  of the submission) and `cross_architecture_section.md` (the full §3.3
  write-up, referenced from the main draft).
- `code/` — all analysis and experiment scripts, numbered roughly in
  pipeline/chronological order. **The numbering has gaps** (e.g.
  `18`→`20`, `32`→`37`): earlier-numbered scripts were superseded by later,
  corrected versions during iterative review (documented in Appendix B),
  or belong to stages whose outputs are already cached in `results/*.json`
  and are not part of this repository. `05_run_surface_baseline.py` and
  `05_surface_baseline_classifier.py` intentionally share a number — the
  former is a thin driver that imports the latter as a helper module
  (explained in the driver's own docstring).
- `results/*.json` — every cached result underlying every number reported in
  the paper. These are sufficient to re-verify all reported statistics
  without re-running any GPU computation.
- `kaggle_kernels/` — every Kaggle GPU run this paper depends on, one
  subdirectory per kernel (script + `kernel-metadata.json` + downloaded
  output). `KAGGLE_RUN_INSTRUCTIONS.md` is an early planning note covering
  only the original cross-architecture replication (Pythia/Qwen0.5B
  extraction) — see this directory itself for the complete, current set of
  runs, including the causal-patching and judge-relabeling kernels added
  during later review rounds.
- `related_work/`, `requirements.txt`.

## What's excluded, and why
Raw per-sample hidden-state feature caches (`results/cross_arch_raw_features_*.npz`,
tens to ~100MB each) are excluded from this repository. These are intermediate
GPU-extraction outputs, not final results — every statistic computed from them is
already saved in the small `results/*.json` files above. A few later-stage scripts
(e.g. `07_multi_arch_causal_patch.py`, `11_multi_arch_difficulty_matched_control.py`)
read these `.npz` caches as input; to regenerate them, run the earlier
feature-extraction stage first (`02_cross_arch_component_probe.py`, which writes
the `.npz` file as one of its outputs) — this requires GPU access and the
relevant model weights, which no cloned repo can shortcut regardless of file
inclusion.

Four scripts (`code/01_ffn_causal_patch.py`, `code/03_fisher_geometry_ffn_attn.py`,
`code/06_difficulty_matched_control.py`, `code/15_sae_feature_gating_utility.py`)
additionally depend on unshipped sibling projects (`mech-int`, `geom-proof`) for
full from-scratch reproduction — disclosed in the paper's "Data and code
availability" section.

## Verifying a reported number without GPU access
Every number in the paper traces to a small, included `results/*.json` file. Open
the relevant JSON directly rather than re-running the pipeline.
