"""Plot RQ2 Pareto frontier: accuracy vs access cost.

Left panel: 50q factual benchmark — FA (y) vs mean probes/query (x).
Right panel: 30q privacy distractor — FA (y) vs forbidden-tool
violation rate (x).

Saved to paper/figures/rq2-pareto.png. Used to make concrete the paper's
"AURA is not the raw-FA winner but the access Pareto point" claim.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from statistics import mean
from typing import Dict

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "evaluation" / "results"
OUT = ROOT / "paper" / "figures" / "rq2-pareto.png"


def _fa_mean(seed_block, baseline_key: str) -> float:
    fa = seed_block.get(baseline_key, {}).get("avg_factual_accuracy")
    return float(fa) if fa is not None else 0.0


def _probes_mean(seed_block, baseline_key: str) -> float:
    details = seed_block.get(baseline_key, {}).get("details", [])
    if not details:
        return 0.0
    vs = [d.get("tool_calls", 0) for d in details]
    return float(mean(vs)) if vs else 0.0


def load_rq2_panel() -> Dict[str, Dict[str, float]]:
    """Aggregate FA and probes/query for the 50q factual benchmark."""
    fa_file = json.load(open(RESULTS / "rq2_factual_accuracy_multiseed.json"))
    extras = json.load(open(RESULTS / "rq2_extra_baselines_multiseed.json"))
    fixed = json.load(open(RESULTS / "rq2_fixed_probe_multiseed.json"))
    gap = json.load(open(RESULTS / "rq2_aura_gap_routed_multiseed.json"))

    seeds = ["42", "123", "456"]
    panel: Dict[str, Dict[str, float]] = {}

    for cond in ["Vanilla_LLM", "Static_Context", "ReAct", "AURA_Full", "AURA_NoProbe"]:
        fas = [_fa_mean(fa_file["per_seed"][s], cond) for s in seeds]
        # AURA_Full uses server-side probes; record the configured cap 2 as
        # an approximation since per-detail probe counts aren't logged for it
        probes = (
            [_probes_mean(fa_file["per_seed"][s], cond) for s in seeds]
            if cond in {"ReAct"}
            else [0.0, 0.0, 0.0]
        )
        if cond == "AURA_Full":
            probes = [2.0, 2.0, 2.0]  # configured probe_max_steps
        panel[cond] = {"fa": mean(fas), "probes": mean(probes)}

    for cond in ["Plan_and_Solve", "Reflexion"]:
        fas = [_fa_mean(extras["per_seed"][s], cond) for s in seeds]
        probes = [_probes_mean(extras["per_seed"][s], cond) for s in seeds]
        panel[cond] = {"fa": mean(fas), "probes": mean(probes)}

    panel["Fixed_Probe"] = {
        "fa": fixed["multi_seed_summary"]["Fixed_Probe"]["fa_mean"],
        "probes": 8.0,
    }
    panel["AURA_GapRouted"] = {
        "fa": gap["multi_seed_summary"]["AURA_GapRouted"]["fa_mean"],
        "probes": gap["multi_seed_summary"]["AURA_GapRouted"]["probes_mean"],
    }
    return panel


def load_privacy_panel() -> Dict[str, Dict[str, float]]:
    data = json.load(open(RESULTS / "rq2_privacy_distractor_multiseed.json"))
    out: Dict[str, Dict[str, float]] = {}
    for name, rec in data.get("multi_seed_summary", {}).items():
        out[name] = {
            "fa": rec["fa_mean"],
            "viol": rec["viol_rate_mean"],
            "probes": rec.get("probes_mean", 0.0),
        }
    return out


def _pareto_front_xy_lower_x(points):
    """Return points on the Pareto frontier minimising x (cost) and
    maximising y (accuracy). points is iterable of (x, y, name)."""
    pts = sorted(points, key=lambda p: (p[0], -p[1]))
    out = []
    best_y = -1.0
    for x, y, name in pts:
        if y > best_y:
            out.append((x, y, name))
            best_y = y
    return out


def main() -> int:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed", file=sys.stderr)
        return 2

    rq2 = load_rq2_panel()
    priv = load_privacy_panel()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(11.5, 4.6))

    # ── Left panel: RQ2 factual (FA vs probes) ─────────────────────────────
    aura_key = "AURA_GapRouted"
    palette = {
        "Vanilla_LLM": ("#888888", "o"),
        "Static_Context": ("#888888", "s"),
        "AURA_NoProbe": ("#888888", "D"),
        "ReAct": ("#3a7ca5", "o"),
        "Reflexion": ("#3a7ca5", "v"),
        "AURA_Full": ("#3a7ca5", "P"),
        "Plan_and_Solve": ("#3a7ca5", "X"),
        "Fixed_Probe": ("#c03434", "^"),
        "AURA_GapRouted": ("#d4a017", "*"),
    }
    pareto_xy = []
    label_offsets_l = {
        "Vanilla_LLM":     (10, -2,  "left"),
        "Static_Context":  (10, -2,  "left"),
        "AURA_NoProbe":    (10,  8,  "left"),
        "ReAct":           (10, -10, "left"),
        "Reflexion":       (10,  0,  "left"),
        "AURA_Full":       (10, 10,  "left"),
        "AURA_GapRouted":  (-10, 10, "right"),
        "Plan_and_Solve":  (8,   8,  "left"),
        "Fixed_Probe":     (-8,  8,  "right"),
    }
    for name, rec in rq2.items():
        color, marker = palette.get(name, ("#444", "o"))
        size = 320 if name == aura_key else (180 if name == "Fixed_Probe" else 110)
        ax_l.scatter(rec["probes"], rec["fa"], c=color, marker=marker, s=size,
                     edgecolors="black", linewidths=0.6, zorder=3,
                     label=name.replace("_", " "))
        dx, dy, ha = label_offsets_l.get(name, (8, 4, "left"))
        ax_l.annotate(
            name.replace("_LLM", "").replace("_", " "),
            (rec["probes"], rec["fa"]),
            textcoords="offset points",
            xytext=(dx, dy),
            fontsize=8,
            ha=ha,
            zorder=4,
        )
        pareto_xy.append((rec["probes"], rec["fa"], name))

    front = _pareto_front_xy_lower_x(pareto_xy)
    ax_l.plot([p[0] for p in front], [p[1] for p in front],
              ls="--", color="#666", lw=1.0, zorder=2, label="Pareto frontier")

    ax_l.set_xlabel("Mean probes per query (lower = cheaper access)")
    ax_l.set_ylabel("Factual accuracy")
    ax_l.set_title("Factual lookup (n=50 × 3 seeds)")
    ax_l.grid(True, ls=":", lw=0.5, alpha=0.5)
    ax_l.set_xlim(-0.5, 9)
    ax_l.set_ylim(0, 1.0)

    # ── Right panel: privacy slice (FA vs violation rate) ─────────────────
    pareto_xy_p = []
    label_offsets_r = {
        "Vanilla_LLM":     (10, -2,  "left"),
        "Static_Context":  (10, -10, "left"),
        "AURA_GapRouted":  (10,  10, "left"),
        "ReAct":           (10,  8,  "left"),
        "Plan_and_Solve":  (-10, -10, "right"),
        "Fixed_Probe":     (-10,  8,  "right"),
    }
    for name, rec in priv.items():
        color, marker = palette.get(name, ("#444", "o"))
        size = 320 if name == aura_key else (180 if name == "Fixed_Probe" else 110)
        ax_r.scatter(rec["viol"], rec["fa"], c=color, marker=marker, s=size,
                     edgecolors="black", linewidths=0.6, zorder=3)
        dx, dy, ha = label_offsets_r.get(name, (8, 6, "left"))
        ax_r.annotate(
            name.replace("_LLM", "").replace("_", " "),
            (rec["viol"], rec["fa"]),
            textcoords="offset points",
            xytext=(dx, dy),
            fontsize=8,
            zorder=4,
            ha=ha,
        )
        pareto_xy_p.append((rec["viol"], rec["fa"], name))

    front_p = _pareto_front_xy_lower_x(pareto_xy_p)
    ax_r.plot([p[0] for p in front_p], [p[1] for p in front_p],
              ls="--", color="#666", lw=1.0, zorder=2)

    ax_r.set_xlabel("Forbidden-tool violation rate (lower = more selective)")
    ax_r.set_ylabel("Factual accuracy")
    ax_r.set_title("Privacy distractor (n=30 × 3 seeds)")
    ax_r.grid(True, ls=":", lw=0.5, alpha=0.5)
    ax_r.set_xlim(-0.05, 1.05)
    ax_r.set_ylim(0, 1.0)

    fig.suptitle("AURA GapRouted on the access-cost Pareto frontier",
                 fontsize=11, y=1.005)
    fig.tight_layout()
    fig.savefig(OUT, dpi=160, bbox_inches="tight")
    print(f"saved → {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
