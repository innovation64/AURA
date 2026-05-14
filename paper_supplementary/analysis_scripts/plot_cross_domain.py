"""Grouped bar chart of cross-domain results: Puzzle / Math / Trust / MemoryArena / GAIA.

Each domain has different metric units; we normalise per-domain to [0, 1] of
the best result and annotate the raw value above each bar so the bars are
visually comparable while preserving the original numbers.
"""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Hand-curated from the appendix tables — single source of truth here.
# format: domain → list of (system, raw_value, metric_label, lower_better?)
DATA = {
    'Puzzle\n(20q, accuracy)':         [('Vanilla',  0.65, 'acc'),
                                        ('AURA no-probe', 0.45, 'acc'),
                                        ('AURA full', 0.70, 'acc')],
    'Math\n(20q, accuracy)':           [('Vanilla',  0.000, 'acc'),
                                        ('AURA no-probe', 0.019, 'acc'),
                                        ('AURA full', 0.019, 'acc')],
    'Trust\n(payoff/round, 6 games)':  [('Vanilla',  1.31, 'payoff/rd'),
                                        ('AURA no-probe', 1.64, 'payoff/rd'),
                                        ('AURA full', 1.15, 'payoff/rd')],
    'MemoryArena\n(1 task, PS)':       [('AURA mem-only', 1.0, 'PS'),
                                        ('AURA no-probe', 1.0, 'PS'),
                                        ('AURA full', 0.4, 'PS')],
    'GAIA\n(139q, accuracy)':          [('direct', 0.094, 'acc'),
                                        ('probe (Explore)', 0.086, 'acc')],
}

domains = list(DATA.keys())
n_dom = len(domains)

fig, ax = plt.subplots(figsize=(12, 4.6))
bar_w = 0.20

palette = {
    'Vanilla': '#94a3b8',
    'direct':  '#94a3b8',
    'AURA no-probe':   '#fb923c',
    'AURA mem-only':   '#fcd34d',
    'AURA full':       '#2563eb',
    'probe (Explore)': '#2563eb',
}

# Per-domain x positions
xpos = np.arange(n_dom) * 1.6
for di, (dom, rows) in enumerate(DATA.items()):
    n_bars = len(rows)
    offsets = (np.arange(n_bars) - (n_bars - 1) / 2) * (bar_w + 0.05)
    for bi, (sys, val, _) in enumerate(rows):
        x = xpos[di] + offsets[bi]
        color = palette.get(sys, '#cbd5e1')
        bar = ax.bar(x, val, width=bar_w, color=color,
                     edgecolor='white', linewidth=1.0, label=sys if di == 0 else None)
        ax.annotate(f'{val:.2f}',
                    xy=(x, val), xytext=(0, 4),
                    textcoords='offset points', ha='center',
                    fontsize=8.5, color='#1e293b')

# X-axis: one tick per domain
ax.set_xticks(xpos)
ax.set_xticklabels(domains, fontsize=9.5)
ax.set_ylabel('Score (raw, native metric per domain)', fontsize=10)
ax.set_ylim(0, max(2.0, max([max(r[1] for r in rows) for rows in DATA.values()]) * 1.18))

# Domain dividers
for x in xpos[1:]:
    ax.axvline(x - 0.8, color='#e2e8f0', linewidth=0.7, zorder=0)

# Legend with all unique systems
from matplotlib.patches import Patch
order = ['Vanilla', 'direct', 'AURA no-probe', 'AURA mem-only', 'AURA full', 'probe (Explore)']
seen = set()
handles = []
for s in order:
    if s in seen:
        continue
    if any(s in [r[0] for r in rows] for rows in DATA.values()):
        handles.append(Patch(facecolor=palette[s], edgecolor='white', label=s))
        seen.add(s)
ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.18),
          ncol=len(handles), frameon=False, fontsize=9)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(axis='y', labelsize=9)
ax.set_axisbelow(True)
ax.grid(axis='y', linestyle=':', color='#e2e8f0')

ax.set_title('Cross-domain results (Appendices J–K). Bar height = raw metric (native units per domain).\n'
             'AURA full vs Vanilla / direct: positive on Puzzle (+5pp), tied/null on Math, '
             'negative on Trust + MemoryArena + GAIA — the empirical envelope where the probing mechanism transfers.',
             fontsize=10, weight='normal')

OUT = ROOT / 'paper' / 'figures' / 'cross-domain.png'
fig.savefig(OUT, dpi=180, bbox_inches='tight', facecolor='white')
print(f'Wrote {OUT}  ({OUT.stat().st_size // 1024} KB)')
