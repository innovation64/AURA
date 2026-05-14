"""Plot the per-rater Δ (AURA - Vanilla) distribution for each of the 4
RQ5 dimensions as a violin + strip plot. Output PNG goes into
paper/figures/ and is wired into the LaTeX paper.
"""
import json, glob
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / 'evaluation' / 'results'
ANN  = RESULTS / 'annotations'
SCEN = RESULTS / 'human_eval_forms.json'
OUT  = ROOT / 'paper' / 'figures' / 'rq5-violin.png'

DIMS = ['response_helpfulness', 'environmental_awareness',
        'agent_believability', 'factual_accuracy']
LABELS = ['Helpfulness', 'Env Awareness', 'Believability', 'Factual Accuracy']

scen = json.loads(SCEN.read_text())
sid_label = {s['id']: ('a' if s['_label_a'] == 'aura' else 'b') for s in scen}

raters = {}
for fp in sorted(ANN.glob('*.json')):
    d = json.loads(fp.read_text())
    raters[d['annotator_id']] = d['ratings']
rids = sorted(raters)
print(f'Loaded {len(rids)} raters')

# For each dim, collect per-rater list of Δ values across 50 scenarios
per_rater_diffs = {dim: {r: [] for r in rids} for dim in DIMS}
for r in rids:
    for sid in range(50):
        a_side = sid_label[sid]; b_side = 'b' if a_side == 'a' else 'a'
        for dim in DIMS:
            va = raters[r].get(f's{sid}_{a_side}_{dim}')
            vb = raters[r].get(f's{sid}_{b_side}_{dim}')
            if va is not None and vb is not None:
                per_rater_diffs[dim][r].append(va - vb)

# Figure: 1 row × 4 dims; each panel is violin of pooled Δ + strip of per-rater means
fig, axes = plt.subplots(1, 4, figsize=(13, 4.0), sharey=True)
fig.subplots_adjust(left=0.06, right=0.99, top=0.86, bottom=0.20, wspace=0.10)

palette = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6',
           '#06b6d4', '#ec4899', '#84cc16']  # 8 colors

for ax, dim, label in zip(axes, DIMS, LABELS):
    pooled = [d for r in rids for d in per_rater_diffs[dim][r]]
    # Violin of pooled Δ
    parts = ax.violinplot([pooled], positions=[0], widths=0.95,
                          showmeans=False, showmedians=False, showextrema=False)
    for pc in parts['bodies']:
        pc.set_facecolor('#cbd5e1')
        pc.set_edgecolor('#475569')
        pc.set_alpha(0.55)

    # Per-rater means as colored dots scattered around x=0
    rng = np.random.default_rng(42)
    for ri, r in enumerate(rids):
        m = np.mean(per_rater_diffs[dim][r]) if per_rater_diffs[dim][r] else 0
        x = rng.uniform(-0.18, 0.18)
        ax.scatter([x], [m], s=110, color=palette[ri % len(palette)],
                   edgecolor='white', linewidth=1.5, zorder=5)
        # Label rater initial near point
        ax.annotate(r[:1] if r != 'Mengting Jia' else 'M',
                    xy=(x, m), xytext=(0, 6),
                    textcoords='offset points', ha='center',
                    fontsize=8, color='#1e293b')

    # Zero line
    ax.axhline(0, color='#94a3b8', linestyle='--', linewidth=0.8, zorder=2)
    # Pooled mean (black diamond)
    pooled_mean = np.mean(pooled)
    ax.scatter([0], [pooled_mean], marker='D', s=80, color='black',
               zorder=6, label=f'pooled $\\Delta$ = {pooled_mean:+.2f}')

    ax.set_title(label, fontsize=11, weight='bold')
    ax.set_xlim(-0.6, 0.6)
    ax.set_xticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.18),
              frameon=False, fontsize=8.5)

axes[0].set_ylabel(r'$\Delta$ = AURA - Vanilla (5-point Likert)', fontsize=10)
axes[0].set_ylim(-2.5, 4.5)
axes[0].axhspan(-2.5, 0, alpha=0.05, color='#dc2626')   # Vanilla-favoring half
axes[0].axhspan(0, 4.5, alpha=0.05, color='#16a34a')    # AURA-favoring half
for ax in axes[1:]:
    ax.axhspan(-2.5, 0, alpha=0.05, color='#dc2626')
    ax.axhspan(0, 4.5, alpha=0.05, color='#16a34a')

fig.suptitle(f'RQ5: per-rater $\\Delta$ (AURA - Vanilla) across 4 dimensions, '
             f'$N$={len(rids)} raters, 50 scenarios each',
             fontsize=11.5, y=0.985)

OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=180, bbox_inches='tight', facecolor='white')
print(f'Wrote {OUT}')
print(f'  {OUT.stat().st_size // 1024} KB')
