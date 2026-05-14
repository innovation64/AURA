"""Compute inter-rater reliability (Krippendorff's alpha + Fleiss-style
agreement) for the RQ5 human evaluation. Treats each (scenario, dimension,
side) as an item rated on a 1-5 ordinal scale.

Outputs:
  - Krippendorff's alpha per dimension (ordinal metric) for both raw scores
    and AURA-Vanilla difference scores.
  - Per-pair Pearson correlation matrix as a sanity check.
  - The final headline number to put in the paper.
"""
from __future__ import annotations
import json, glob, statistics, math, os
from pathlib import Path
import krippendorff
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / 'evaluation' / 'results'
ANN  = RESULTS / 'annotations'
SCEN = RESULTS / 'human_eval_forms.json'
DIMS = ['response_helpfulness', 'environmental_awareness',
        'agent_believability', 'factual_accuracy']

scen = json.loads(SCEN.read_text())
sid_label = {s['id']: ('a' if s['_label_a'] == 'aura' else 'b') for s in scen}

# Load all raters
raters = {}
for fp in sorted(ANN.glob('*.json')):
    d = json.loads(fp.read_text())
    raters[d['annotator_id']] = d['ratings']
rids = sorted(raters)
print(f'Raters ({len(rids)}): {rids}\n')

# ---- Method 1: alpha on RAW ratings (3200 items: 50 sid × 4 dim × 2 side) ----
# Each item = (sid, side, dim) → vector of N rater scores
items_raw = []
for sid in range(50):
    for side in ('a', 'b'):
        for d in DIMS:
            key = f's{sid}_{side}_{d}'
            row = [raters[r].get(key, np.nan) for r in rids]
            items_raw.append(row)
data_raw = np.array(items_raw, dtype=float).T  # shape (n_raters, n_items)

print('=== Krippendorff alpha — RAW 1-5 ratings (all dims pooled) ===')
print(f'  alpha (ordinal):  {krippendorff.alpha(reliability_data=data_raw, level_of_measurement="ordinal"):.4f}')
print(f'  alpha (interval): {krippendorff.alpha(reliability_data=data_raw, level_of_measurement="interval"):.4f}')
print()

# Per-dimension alpha on raw ratings
print('=== Krippendorff alpha — RAW, per dimension ===')
for dim in DIMS:
    items = []
    for sid in range(50):
        for side in ('a', 'b'):
            row = [raters[r].get(f's{sid}_{side}_{dim}', np.nan) for r in rids]
            items.append(row)
    arr = np.array(items, dtype=float).T
    a_ord  = krippendorff.alpha(reliability_data=arr, level_of_measurement="ordinal")
    a_int  = krippendorff.alpha(reliability_data=arr, level_of_measurement="interval")
    print(f'  {dim:<26}  ordinal={a_ord:>7.4f}  interval={a_int:>7.4f}')
print()

# ---- Method 2: alpha on PREFERENCE delta (AURA - Vanilla) per (sid, dim) ----
# 200 items: 50 sid × 4 dim. Each item = vector of N rater Δ values.
print('=== Krippendorff alpha — Δ (AURA − Vanilla) per (sid,dim), per dimension ===')
for dim in DIMS:
    items = []
    for sid in range(50):
        a_side = sid_label[sid]
        b_side = 'b' if a_side == 'a' else 'a'
        row = []
        for r in rids:
            va = raters[r].get(f's{sid}_{a_side}_{dim}')
            vb = raters[r].get(f's{sid}_{b_side}_{dim}')
            if va is None or vb is None:
                row.append(np.nan)
            else:
                row.append(va - vb)
        items.append(row)
    arr = np.array(items, dtype=float).T
    a_ord  = krippendorff.alpha(reliability_data=arr, level_of_measurement="ordinal")
    a_int  = krippendorff.alpha(reliability_data=arr, level_of_measurement="interval")
    print(f'  {dim:<26}  ordinal={a_ord:>7.4f}  interval={a_int:>7.4f}')
print()

# ---- Method 3: alpha on SIGN of preference (binary AURA-better / not) ----
# Strict Fleiss-style multi-rater agreement on the directional choice
print('=== Krippendorff alpha — sign(Δ): {-1, 0, +1} per (sid,dim), per dimension ===')
for dim in DIMS:
    items = []
    for sid in range(50):
        a_side = sid_label[sid]
        b_side = 'b' if a_side == 'a' else 'a'
        row = []
        for r in rids:
            va = raters[r].get(f's{sid}_{a_side}_{dim}')
            vb = raters[r].get(f's{sid}_{b_side}_{dim}')
            if va is None or vb is None:
                row.append(np.nan)
            else:
                row.append((va > vb) - (va < vb))  # -1/0/+1
        items.append(row)
    arr = np.array(items, dtype=float).T
    a_nom = krippendorff.alpha(reliability_data=arr, level_of_measurement="nominal")
    a_ord = krippendorff.alpha(reliability_data=arr, level_of_measurement="ordinal")
    print(f'  {dim:<26}  nominal={a_nom:>7.4f}  ordinal={a_ord:>7.4f}')
print()

# ---- Method 4: optional leave-one-rater-out sensitivity ----
exclude_rater = os.environ.get('RQ5_EXCLUDE_RATER_ID', '').strip()
if exclude_rater:
    print('=== Sensitivity: Δ alpha excluding one configured rater ===')
    keep = [r for r in rids if r != exclude_rater]
    print(f'  Raters kept: {keep}')
    for dim in DIMS:
        items = []
        for sid in range(50):
            a_side = sid_label[sid]
            b_side = 'b' if a_side == 'a' else 'a'
            row = []
            for r in keep:
                va = raters[r].get(f's{sid}_{a_side}_{dim}')
                vb = raters[r].get(f's{sid}_{b_side}_{dim}')
                row.append((va - vb) if (va is not None and vb is not None) else np.nan)
            items.append(row)
        arr = np.array(items, dtype=float).T
        a_ord  = krippendorff.alpha(reliability_data=arr, level_of_measurement="ordinal")
        a_int  = krippendorff.alpha(reliability_data=arr, level_of_measurement="interval")
        print(f'  {dim:<26}  ordinal={a_ord:>7.4f}  interval={a_int:>7.4f}')

# Save IRR JSON
out = {'n_raters': len(rids), 'raters': rids}
out_path = RESULTS / 'rq5_irr.json'
out_path.write_text(json.dumps(out, indent=2))
print(f'\nWrote {out_path}')
