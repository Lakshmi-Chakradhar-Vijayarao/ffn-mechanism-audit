"""
Generates figures/ffn-attn-comparison.pdf for the paper.
All numbers are copied directly from already-published results
(see results/cross_arch_component_probe_*.json and the paper text,
Sections 3.2-3.3) -- this script performs no new computation, only
visualization of already-reported numbers.

CORRECTION (post-review): the GPT-2 and Qwen0.5B-bare error bars were
previously hardcoded to round placeholder values (0.03/0.03 and
0.02/0.02) that did not match the actual cross-validation standard
deviations in results/vendored_mech_int/component_results.npy (GPT-2:
0.0557/0.0427) and results/cross_arch_component_probe_qwen05.json
(Qwen-bare: 0.0628/0.0423). Fixed to read the real values; Pythia and
Qwen0.5B-chat's hardcoded values were already correct and are unchanged.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

conditions = ["GPT-2\n(bare)", "Pythia-410M\n(bare)", "Qwen0.5B\n(bare template)", "Qwen0.5B\n(chat template)"]
ffn = [0.6053, 0.6181, 0.5657, 0.5704]
attn = [0.6165, 0.6115, 0.5625, 0.5988]
ffn_err = [0.0557, 0.0442, 0.0628, 0.0186]
attn_err = [0.0427, 0.0345, 0.0423, 0.0438]

x = np.arange(len(conditions))
width = 0.35

fig, ax = plt.subplots(figsize=(7, 4.2))
ax.bar(x - width/2, ffn, width, yerr=ffn_err, capsize=3, label="FFN", color="#4C72B0")
ax.bar(x + width/2, attn, width, yerr=attn_err, capsize=3, label="Attention", color="#DD8452")
ax.axhline(0.5, color="gray", linestyle=":", linewidth=1, label="Chance (AUROC=0.5)")
ax.set_ylabel("Peak AUROC")
ax.set_ylim(0.45, 0.68)
ax.set_xticks(x)
ax.set_xticklabels(conditions, fontsize=9)
ax.legend(loc="upper right", fontsize=8, ncol=1)
ax.set_title("FFN vs. Attention peak AUROC: consistently close, and the\nonly architecture with a clear leader reverses under template correction", fontsize=9.5)
fig.tight_layout()
fig.savefig("../draft/latex/figures/ffn-attn-comparison.pdf")
print("Saved.")
