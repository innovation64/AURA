"""SOTOPIA 7-dimension radar plot for RQ4.

Each dimension has a different native scale; we normalise each axis to [0, 1]
with the dimension's bound visible on the radial label, then annotate the
raw score next to each point so readers can read both views.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# RQ4 SOTOPIA results from the AURA full 200-step run (paper Table 8).
# Each tuple: (label, raw_value, low, high)  with native [low, high] range.
DIMS = [
    ('Believability',  9.0,    0,  10),
    ('Goal',           9.5,    0,  10),
    ('Knowledge',      8.0,    0,  10),
    ('Relationship',   2.07,  -5,   5),
    ('Financial',      0.63,  -5,   5),
    ('Secret',        -0.67, -10,   0),
    ('Social Rules',  -2.13, -10,   0),
]

labels = [d[0] for d in DIMS]
raw    = [d[1] for d in DIMS]
lows   = [d[2] for d in DIMS]
highs  = [d[3] for d in DIMS]
# Normalise each axis to [0, 1]
norm = [(r - l) / (h - l) for r, l, h in zip(raw, lows, highs)]

# Close the polygon
N = len(DIMS)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
norm_closed   = norm + [norm[0]]
angles_closed = angles + [angles[0]]

fig, ax = plt.subplots(figsize=(7.0, 6.5), subplot_kw={'projection': 'polar'})
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)

# Plot AURA
ax.plot(angles_closed, norm_closed, color='#2563eb', linewidth=2.0, label='AURA full (200 sim steps)')
ax.fill(angles_closed, norm_closed, color='#2563eb', alpha=0.15)
# Scatter + annotate raw values
for ang, n, r, lo, hi, lab in zip(angles, norm, raw, lows, highs, labels):
    ax.scatter([ang], [n], color='#2563eb', s=70, zorder=5, edgecolor='white', linewidth=1.5)
    sign = '+' if r > 0 else ''
    ax.annotate(f'{sign}{r:.2f}',
                xy=(ang, n), xytext=(ang, n + 0.10),
                ha='center', fontsize=9, color='#1e40af', weight='bold')

# Axis labels show "Dim\n(low ↔ high)"
axis_labels = [f'{lab}\n({lo}…{hi})' for lab, lo, hi in zip(labels, lows, highs)]
ax.set_xticks(angles)
ax.set_xticklabels(axis_labels, fontsize=10)

ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
ax.set_yticklabels(['0', '.25', '.5', '.75', '1.0'], fontsize=8, color='#94a3b8')
ax.set_ylim(0, 1.05)
ax.spines['polar'].set_color('#cbd5e1')

# Reference: a "neutral" performance line at 0.5 (mid-scale on every axis)
neutral = [0.5] * (N + 1)
ax.plot(angles_closed, neutral, color='#94a3b8', linestyle=':', linewidth=0.9, alpha=0.7,
        label='neutral (mid-scale)')

ax.set_title('RQ4: SOTOPIA 7-dimension social evaluation (AURA, 200-step run)\n'
             'Each axis normalised to [0,1] of its native range; raw values annotated',
             fontsize=10, pad=22, weight='bold')
ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.18),
          ncol=2, frameon=False, fontsize=9)

OUT = ROOT / 'paper' / 'figures' / 'rq4-sotopia-radar.png'
fig.savefig(OUT, dpi=180, bbox_inches='tight', facecolor='white')
print(f'Wrote {OUT}  ({OUT.stat().st_size // 1024} KB)')
