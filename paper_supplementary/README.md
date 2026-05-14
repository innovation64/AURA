# Paper Supplementary

Companion artefacts for the paper *AURA: Intent-Directed Probing for
Implicit-Need Surfacing in Situated LLM Agents*. The AURA framework code lives
in `../src/aura/`; this directory contains the simulator, benchmarks, IAA
materials, and per-seed run records released alongside the paper.

## Directory layout

| Subdir | Contents |
|---|---|
| `town_simulator/` | The AURATown simulator (Python) — 60×60 grid, 5 agents, 20 named locations, 7-rule deterministic private-state evolution, scripted probe-tool registry. Self-contained; runs without an LLM via the heuristic backend. |
| `visualization_ui/` | React/Vite web UI for the simulator (chat view, agent panel, map, event log). Talks to the Python HTTP API in `town_simulator/server.py`. |
| `benchmarks/` | All datasets evaluated in the paper: `implicit_intent_queries.json` (the primary 25-query benchmark, 5 subcategories), `implicit_intent_queries_v2.json` (the 100-query four-scene expansion), `environment_grounding_tests.json` (50-query factual benchmark), `privacy_distractor_queries.json` (30-query privacy slice), and `chat_queries.json` (action-grounding workload). |
| `iaa/` | Inter-annotator-agreement materials for the subcategory labels: two annotators (`annotator_1`, `annotator_2`) independently labelled the 25 implicit-intent queries; `iaa_form.html` is the form they filled, `score_iaa.py` recomputes Cohen's κ. |
| `results/` | Per-seed run records for every condition reported in the paper (JSON; one file per experiment × seed where applicable). `results/annotations/` contains the 8-rater RQ5 human eval, with rater IDs anonymised to `annotator_1`…`annotator_8`. |
| `eval_scripts/` | The Python scripts that run each benchmark and re-aggregate results from `results/*.json` (e.g. `run_experiments.py`, `fantom_eval.py`, `gaia_eval.py`, `human_eval_server.py`). |
| `analysis_scripts/` | Plotting + statistical aggregation scripts referenced by the paper figures and tables (e.g. `plot_rq2_pareto.py`, `aggregate_rq2_multiseed.py`, `compute_irr.py`). |

## Anonymisation

All annotator and rater identifiers in `iaa/` and `results/annotations/` have
been replaced with `annotator_N` aliases. The mapping is stable across files
(e.g. `annotator_4` is the same person in every JSON they appear in), so
cross-references in the data remain consistent.

No real-name, e-mail, institution, or personally-identifying path is shipped
in this archive.

## Reproducing the paper

The headline numbers in the paper can be reproduced via:

```bash
# Set OPENAI_API_KEY (or compatible) in your environment first
cd ../../
python -m pip install -e .                            # install the AURA framework

# Primary 25-query implicit-intent benchmark (paper §5.2)
python paper_supplementary/eval_scripts/run_experiments.py \
    --bench paper_supplementary/benchmarks/implicit_intent_queries.json \
    --seeds 42 123 456 \
    --out paper_supplementary/results/

# 100-query v2 expansion (paper §5.2)
python paper_supplementary/eval_scripts/run_experiments.py \
    --bench paper_supplementary/benchmarks/implicit_intent_queries_v2.json \
    --out paper_supplementary/results/

# 50-query factual-grounding benchmark (paper §5.1)
python paper_supplementary/eval_scripts/environment_eval.py \
    --bench paper_supplementary/benchmarks/environment_grounding_tests.json \
    --seeds 42 123 456 \
    --out paper_supplementary/results/

# Aggregate + regenerate paper tables
python paper_supplementary/analysis_scripts/aggregate_rq2_multiseed.py
python paper_supplementary/analysis_scripts/compute_irr.py
```

`results/` is shipped pre-populated with the seed=42/123/456 runs used in the
paper, so plotting + table-generation scripts can be re-run without re-executing
the full benchmark.

## Simulator

The simulator can be launched standalone (no LLM API key needed for the
heuristic-backend smoke tests):

```bash
python -m town_simulator.server          # HTTP API on :7861
cd visualization_ui && npm run dev       # React UI on :5173 (proxies /api to :7861)
```

Browse `http://localhost:5173/` to step through the 35-tick day and inspect
agent public/private state evolution under the 7-rule transition table.

## License

Inherits the MIT license of the parent AURA framework
(`../LICENSE`). The benchmarks are also released under MIT for academic use.
