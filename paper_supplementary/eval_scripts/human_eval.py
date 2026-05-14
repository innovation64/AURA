"""
Human evaluation framework for AURA paper.

Provides:
1. HumanEvalGenerator - generates paired evaluation scenarios (AURA vs Vanilla)
   as JSON and an HTML form for human annotators.
2. HumanEvalAnalyzer - loads completed annotations, computes inter-rater
   reliability (Krippendorff's alpha, Cohen's kappa) and significance
   (Wilcoxon signed-rank test).
3. regenerate_forms() - re-queries the AURA server for fresh responses using
   the FIXED system prompt, then rebuilds the HTML/JSON forms. Call this
   AFTER the current experiment run finishes (so the server can be reset).

Evaluation dimensions (each 1-5):
  - response_helpfulness
  - environmental_awareness
  - agent_believability
  - factual_accuracy
"""

import html
import json
import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

EVAL_DIMENSIONS = [
    "response_helpfulness",
    "environmental_awareness",
    "agent_believability",
    "factual_accuracy",
]

# Default server URL for the annotation collection server
ANNOTATION_SERVER_URL = "http://0.0.0.0:5050"


# ===========================================================================
# Generator
# ===========================================================================

@dataclass
class EvalScenario:
    scenario_id: int
    query: str
    agent_name: str
    category: str
    response_a: str  # System A (order randomised)
    response_b: str  # System B
    label_a: str     # "aura" or "baseline" (hidden from annotator)
    label_b: str


class HumanEvalGenerator:
    """Generate paired evaluation materials."""

    def __init__(
        self,
        aura_results: List[Dict[str, Any]],
        baseline_results: List[Dict[str, Any]],
        output_dir: str = "evaluation/results",
    ) -> None:
        self.aura = aura_results
        self.baseline = baseline_results
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate(self, seed: int = 42) -> List[EvalScenario]:
        """Create paired evaluation scenarios with randomised presentation order."""
        import random
        rng = random.Random(seed)

        scenarios: List[EvalScenario] = []
        n = min(len(self.aura), len(self.baseline))

        for i in range(n):
            a_resp = self.aura[i].get("response", "")
            b_resp = self.baseline[i].get("response", "")
            query = self.aura[i].get("query", self.baseline[i].get("query", ""))
            agent = self.aura[i].get("agent", "")
            cat = self.aura[i].get("category", "")

            # Randomise order
            if rng.random() < 0.5:
                scenarios.append(EvalScenario(
                    scenario_id=i,
                    query=query,
                    agent_name=agent,
                    category=cat,
                    response_a=a_resp,
                    response_b=b_resp,
                    label_a="aura",
                    label_b="baseline",
                ))
            else:
                scenarios.append(EvalScenario(
                    scenario_id=i,
                    query=query,
                    agent_name=agent,
                    category=cat,
                    response_a=b_resp,
                    response_b=a_resp,
                    label_a="baseline",
                    label_b="aura",
                ))

        return scenarios

    def save_json(self, scenarios: List[EvalScenario]) -> Path:
        """Save scenarios as JSON (for programmatic annotation tools)."""
        data = []
        for s in scenarios:
            data.append({
                "id": s.scenario_id,
                "query": s.query,
                "agent": s.agent_name,
                "category": s.category,
                "response_a": s.response_a,
                "response_b": s.response_b,
                # Hidden labels -- remove before sending to annotators
                "_label_a": s.label_a,
                "_label_b": s.label_b,
            })
        path = self.output_dir / "human_eval_forms.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return path

    def save_html(self, scenarios: List[EvalScenario]) -> Path:
        """Generate an HTML evaluation form that submits to the collection server."""
        total = len(scenarios)
        rows_html = ""
        for s in scenarios:
            rows_html += _render_scenario_html(s, total)

        page = _build_full_html(rows_html, total)
        path = self.output_dir / "human_eval_forms.html"
        with open(path, "w", encoding="utf-8") as f:
            f.write(page)
        return path


def _render_scenario_html(s: EvalScenario, total: int) -> str:
    dims_rows = ""
    for dim in EVAL_DIMENSIONS:
        dim_label = dim.replace("_", " ").title()
        radios_a = "".join(
            f'<input type="radio" name="s{s.scenario_id}_a_{dim}" value="{v}" required> {v} '
            for v in range(1, 6)
        )
        radios_b = "".join(
            f'<input type="radio" name="s{s.scenario_id}_b_{dim}" value="{v}" required> {v} '
            for v in range(1, 6)
        )
        dims_rows += f"""<tr>
  <td>{dim_label}</td>
  <td>{radios_a}</td>
  <td>{radios_b}</td>
</tr>
"""

    return f"""<div class="scenario" id="scenario-{s.scenario_id}" data-sid="{s.scenario_id}">
<h3>Scenario {s.scenario_id + 1} <span class="progress-label">/ {total}</span></h3>
<p><b>Query:</b> {html.escape(s.query)}</p>
<p><b>Agent:</b> {html.escape(s.agent_name)} | <b>Category:</b> {html.escape(s.category)}</p>
<div class="responses">
  <div class="resp-box"><h4>Response A</h4><p>{html.escape(s.response_a)}</p></div>
  <div class="resp-box"><h4>Response B</h4><p>{html.escape(s.response_b)}</p></div>
</div>
<table class="dims">
<tr><th>Dimension</th><th>Response A (1-5)</th><th>Response B (1-5)</th></tr>
{dims_rows}
</table>
</div>
"""


def _build_full_html(rows_html: str, total: int) -> str:
    """Build the complete HTML page with server submission, progress tracking,
    and save-and-continue-later via localStorage."""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>AURA Human Evaluation</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; max-width: 960px; margin: 0 auto; padding: 1em; color: #222; }}
h1 {{ text-align: center; }}
.header-bar {{ background: #f0f7ff; border: 1px solid #b3d4fc; border-radius: 8px; padding: 1em 1.5em; margin-bottom: 1.5em; }}
.header-bar label {{ font-weight: 600; }}
.header-bar input[type=text] {{ padding: 6px 10px; font-size: 1rem; border: 1px solid #aaa; border-radius: 4px; width: 240px; margin-left: 8px; }}
.progress-bar-container {{ background: #e0e0e0; border-radius: 6px; height: 24px; margin: 0.8em 0; overflow: hidden; }}
.progress-bar {{ background: #4caf50; height: 100%; border-radius: 6px; transition: width 0.3s; display: flex; align-items: center; justify-content: center; color: #fff; font-size: 0.85rem; font-weight: 600; min-width: 40px; }}
.scenario {{ border: 1px solid #ddd; border-radius: 8px; padding: 1.5em; margin-bottom: 2em; background: #fafafa; }}
.scenario.completed {{ border-color: #4caf50; background: #f6fff6; }}
.scenario h3 {{ margin-top: 0; }}
.progress-label {{ color: #888; font-weight: normal; font-size: 0.85em; }}
.responses {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1em; }}
.resp-box {{ background: #fff; border: 1px solid #ccc; border-radius: 6px; padding: 1em; white-space: pre-wrap; }}
.resp-box h4 {{ margin-top: 0; }}
table.dims {{ width: 100%; border-collapse: collapse; margin-top: 1em; }}
table.dims th, table.dims td {{ border: 1px solid #ccc; padding: 6px 10px; text-align: center; }}
table.dims th {{ background: #f0f0f0; }}
input[type=radio] {{ margin: 0 2px; }}
.btn-row {{ display: flex; gap: 1em; justify-content: center; margin-top: 2em; margin-bottom: 3em; }}
.btn-row button {{ padding: 0.8em 2em; font-size: 1rem; cursor: pointer; border-radius: 6px; border: 1px solid #aaa; }}
#btnSubmit {{ background: #4caf50; color: #fff; border-color: #4caf50; }}
#btnSubmit:hover {{ background: #43a047; }}
#btnSave {{ background: #ff9800; color: #fff; border-color: #ff9800; }}
#btnSave:hover {{ background: #fb8c00; }}
.msg {{ text-align: center; padding: 0.5em; margin-top: 1em; border-radius: 6px; display: none; }}
.msg.success {{ background: #c8e6c9; color: #2e7d32; display: block; }}
.msg.error {{ background: #ffcdd2; color: #c62828; display: block; }}
</style>
</head>
<body>
<h1>AURA Human Evaluation</h1>

<div class="header-bar">
  <label for="annotatorId">Annotator ID:</label>
  <input type="text" id="annotatorId" placeholder="e.g. annotator_01" />
  <span style="margin-left: 1em; color: #666; font-size: 0.9em;">Enter your assigned ID before starting.</span>
</div>

<div class="progress-bar-container">
  <div class="progress-bar" id="progressBar" style="width: 0%;">0 / {total}</div>
</div>

<form id="evalForm">
{rows_html}
</form>

<div class="btn-row">
  <button type="button" id="btnSave" onclick="saveProgress()">Save &amp; Continue Later</button>
  <button type="button" id="btnSubmit" onclick="submitForm()">Submit All Annotations</button>
</div>
<div class="msg" id="statusMsg"></div>

<script>
const TOTAL = {total};
const DIMS = {json.dumps(EVAL_DIMENSIONS)};
const STORAGE_KEY = 'aura_human_eval_progress';

// ---- Progress tracking ----
function countCompleted() {{
    let completed = 0;
    for (let sid = 0; sid < TOTAL; sid++) {{
        let allDone = true;
        for (const dim of DIMS) {{
            const nameA = 's' + sid + '_a_' + dim;
            const nameB = 's' + sid + '_b_' + dim;
            if (!document.querySelector('input[name="' + nameA + '"]:checked') ||
                !document.querySelector('input[name="' + nameB + '"]:checked')) {{
                allDone = false;
                break;
            }}
        }}
        const el = document.getElementById('scenario-' + sid);
        if (allDone) {{
            completed++;
            if (el) el.classList.add('completed');
        }} else {{
            if (el) el.classList.remove('completed');
        }}
    }}
    const pct = Math.round(completed / TOTAL * 100);
    const bar = document.getElementById('progressBar');
    bar.style.width = pct + '%';
    bar.textContent = completed + ' / ' + TOTAL;
    return completed;
}}

// Listen for changes
document.getElementById('evalForm').addEventListener('change', function() {{
    countCompleted();
    autoSaveToLocal();
}});

// ---- localStorage persistence ----
function collectFormData() {{
    const form = document.getElementById('evalForm');
    const data = {{}};
    const radios = form.querySelectorAll('input[type=radio]:checked');
    radios.forEach(function(r) {{ data[r.name] = parseInt(r.value); }});
    return data;
}}

function autoSaveToLocal() {{
    const payload = {{
        annotator_id: document.getElementById('annotatorId').value,
        ratings: collectFormData(),
        saved_at: new Date().toISOString(),
    }};
    localStorage.setItem(STORAGE_KEY, JSON.stringify(payload));
}}

function loadFromLocal() {{
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return;
    try {{
        const payload = JSON.parse(raw);
        if (payload.annotator_id) {{
            document.getElementById('annotatorId').value = payload.annotator_id;
        }}
        if (payload.ratings) {{
            for (const [name, val] of Object.entries(payload.ratings)) {{
                const radio = document.querySelector('input[name="' + name + '"][value="' + val + '"]');
                if (radio) radio.checked = true;
            }}
        }}
        countCompleted();
    }} catch(e) {{ /* ignore corrupt data */ }}
}}

function saveProgress() {{
    const aid = document.getElementById('annotatorId').value.trim();
    if (!aid) {{
        showMsg('Please enter your Annotator ID first.', 'error');
        return;
    }}
    autoSaveToLocal();

    // Also POST partial progress to server
    const data = collectFormData();
    const payload = {{ annotator_id: aid, ratings: data, partial: true }};
    fetch('/submit', {{
        method: 'POST',
        headers: {{ 'Content-Type': 'application/json' }},
        body: JSON.stringify(payload),
    }}).then(r => r.json()).then(resp => {{
        if (resp.ok) {{
            showMsg('Progress saved! You can close this page and return later.', 'success');
        }} else {{
            showMsg('Server save failed, but local save succeeded. Error: ' + (resp.error || 'unknown'), 'error');
        }}
    }}).catch(err => {{
        showMsg('Could not reach server. Progress saved locally.', 'error');
    }});
}}

// ---- Submit ----
function submitForm() {{
    const aid = document.getElementById('annotatorId').value.trim();
    if (!aid) {{
        showMsg('Please enter your Annotator ID before submitting.', 'error');
        return;
    }}
    const completed = countCompleted();
    if (completed < TOTAL) {{
        if (!confirm('You have completed ' + completed + '/' + TOTAL + ' scenarios. Submit anyway?')) {{
            return;
        }}
    }}
    const data = collectFormData();
    const payload = {{ annotator_id: aid, ratings: data, partial: false }};
    fetch('/submit', {{
        method: 'POST',
        headers: {{ 'Content-Type': 'application/json' }},
        body: JSON.stringify(payload),
    }}).then(r => r.json()).then(resp => {{
        if (resp.ok) {{
            showMsg('Annotations submitted successfully! Thank you.', 'success');
            localStorage.removeItem(STORAGE_KEY);
        }} else {{
            showMsg('Submission failed: ' + (resp.error || 'unknown error'), 'error');
        }}
    }}).catch(err => {{
        showMsg('Network error: ' + err + '. Try again or use Save.', 'error');
    }});
}}

function showMsg(text, type) {{
    const el = document.getElementById('statusMsg');
    el.textContent = text;
    el.className = 'msg ' + type;
    setTimeout(function() {{ el.className = 'msg'; }}, 8000);
}}

// ---- Init ----
window.addEventListener('load', function() {{
    loadFromLocal();
}});
</script>
</body>
</html>"""


# ===========================================================================
# Form Regeneration (post-experiment)
# ===========================================================================

def regenerate_forms(
    aura_server: str = "http://127.0.0.1:7861",
    queries_path: str = "evaluation/data/chat_queries.json",
    output_dir: str = "evaluation/results",
    num_queries: int = 50,
    warmup_steps: int = 10,
    seed: int = 42,
):
    """
    Re-generate human evaluation forms by querying the AURA server for fresh
    responses using the FIXED system prompt.  Call this AFTER the current
    experiment finishes so the server can be safely reset.

    This replaces the old forms that were generated with the hallucination-prone
    chat prompt.

    Usage (from project root):
        python -c "from evaluation.human_eval import regenerate_forms; regenerate_forms()"
    """
    import requests
    from pathlib import Path

    print("=" * 60)
    print("Regenerating Human Evaluation Forms (FIXED prompt)")
    print("=" * 60)

    # Load queries
    qpath = Path(queries_path)
    if not qpath.exists():
        # Try relative to this file
        qpath = Path(__file__).parent / "data" / "chat_queries.json"
    with open(qpath) as f:
        query_data = json.load(f)
    queries = query_data["queries"][:num_queries]

    base = aura_server.rstrip("/")

    # Check server health
    try:
        r = requests.get(f"{base}/api/health", timeout=5)
        assert r.json().get("ok"), "Server not healthy"
    except Exception as e:
        print(f"ERROR: Cannot connect to AURA server at {base}: {e}")
        print("Make sure the experiment has finished and the server is available.")
        return None

    # Reset and warm up
    print(f"  Resetting server and warming up ({warmup_steps} steps)...")
    requests.post(f"{base}/api/reset", timeout=30)
    requests.post(f"{base}/api/probe", json={"enabled": True, "max_steps": 2}, timeout=10)
    requests.post(f"{base}/api/ablation", json={"memory_enabled": True, "reflection_enabled": True}, timeout=10)
    for step_i in range(warmup_steps):
        requests.post(f"{base}/api/step", timeout=120)
        time.sleep(0.5)

    # Get agent names
    state = requests.get(f"{base}/api/state", timeout=10).json()["state"]
    agent_names = [a["name"] for a in state["agents"]]

    # Collect AURA responses (with fixed prompt)
    print(f"  Collecting {len(queries)} AURA responses...")
    aura_results = []
    for qi, q in enumerate(queries):
        agent_name = agent_names[qi % len(agent_names)]
        try:
            r = requests.post(
                f"{base}/api/chat",
                json={"user": agent_name, "message": q["query"]},
                timeout=120,
            )
            resp_data = r.json()
            if resp_data.get("ok"):
                aura_results.append({
                    "query": q["query"],
                    "agent": agent_name,
                    "category": q.get("category", ""),
                    "response": resp_data.get("chat", {}).get("ai_response", ""),
                })
            else:
                aura_results.append({"query": q["query"], "agent": agent_name, "category": q.get("category", ""), "response": ""})
        except Exception as e:
            print(f"    WARNING: Query {qi} failed: {e}")
            aura_results.append({"query": q["query"], "agent": agent_name, "category": q.get("category", ""), "response": ""})
        if qi % 10 == 0:
            print(f"    AURA query {qi}/{len(queries)}")
        time.sleep(0.3)

    # Collect vanilla baseline responses
    print(f"  Collecting {len(queries)} Vanilla baseline responses...")
    # Import baseline function
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from evaluation.baselines import vanilla_llm_chat
    from evaluation.config import EvalConfig

    config = EvalConfig()
    baseline_results = []
    for qi, q in enumerate(queries):
        agent_name = agent_names[qi % len(agent_names)]
        try:
            bl = vanilla_llm_chat(config, agent_name, q["query"])
            baseline_results.append({
                "query": q["query"],
                "agent": agent_name,
                "category": q.get("category", ""),
                "response": bl.get("response", ""),
            })
        except Exception as e:
            print(f"    WARNING: Baseline query {qi} failed: {e}")
            baseline_results.append({"query": q["query"], "agent": agent_name, "category": q.get("category", ""), "response": ""})
        if qi % 10 == 0:
            print(f"    Baseline query {qi}/{len(queries)}")
        time.sleep(0.3)

    # Generate new forms
    gen = HumanEvalGenerator(aura_results, baseline_results, output_dir)
    scenarios = gen.generate(seed=seed)
    json_path = gen.save_json(scenarios)
    html_path = gen.save_html(scenarios)

    print(f"\n  JSON saved: {json_path}")
    print(f"  HTML saved: {html_path}")
    print(f"  Total scenarios: {len(scenarios)}")
    print("  DONE. Forms regenerated with fixed prompt.")
    return {"json_path": str(json_path), "html_path": str(html_path), "num_scenarios": len(scenarios)}


# ===========================================================================
# Analyzer
# ===========================================================================

class HumanEvalAnalyzer:
    """Analyse completed human evaluation annotations."""

    def __init__(self, scenarios: List[Dict], annotations: List[Dict]) -> None:
        """
        scenarios: list of scenario dicts with _label_a / _label_b
        annotations: list of dicts from annotators, keyed like
                     s{id}_a_{dim} -> score, s{id}_b_{dim} -> score
        """
        self.scenarios = {s["id"]: s for s in scenarios}
        self.annotations = annotations

    def _collect_scores(self) -> Dict[str, Dict[str, List[float]]]:
        """Collect per-dimension scores separated by system (aura vs baseline)."""
        scores: Dict[str, Dict[str, List[float]]] = {
            "aura": defaultdict(list),
            "baseline": defaultdict(list),
        }
        for ann in self.annotations:
            for sid, scenario in self.scenarios.items():
                for dim in EVAL_DIMENSIONS:
                    key_a = f"s{sid}_a_{dim}"
                    key_b = f"s{sid}_b_{dim}"
                    val_a = ann.get(key_a)
                    val_b = ann.get(key_b)
                    if val_a is not None:
                        scores[scenario["_label_a"]][dim].append(float(val_a))
                    if val_b is not None:
                        scores[scenario["_label_b"]][dim].append(float(val_b))
        return scores

    def compute_means(self) -> Dict[str, Dict[str, float]]:
        """Compute mean scores per system per dimension."""
        scores = self._collect_scores()
        means: Dict[str, Dict[str, float]] = {}
        for system in ["aura", "baseline"]:
            means[system] = {}
            for dim in EVAL_DIMENSIONS:
                vals = scores[system][dim]
                means[system][dim] = round(sum(vals) / max(len(vals), 1), 3) if vals else 0
        return means

    def wilcoxon_test(self) -> Dict[str, Dict[str, Any]]:
        """
        Wilcoxon signed-rank test per dimension (paired: AURA vs Baseline
        on the same scenario).  Returns test statistic and approximate p-value.
        """
        results: Dict[str, Dict[str, Any]] = {}
        for dim in EVAL_DIMENSIONS:
            aura_scores: List[float] = []
            base_scores: List[float] = []

            for ann in self.annotations:
                for sid, scenario in self.scenarios.items():
                    key_a = f"s{sid}_a_{dim}"
                    key_b = f"s{sid}_b_{dim}"
                    val_a = ann.get(key_a)
                    val_b = ann.get(key_b)
                    if val_a is not None and val_b is not None:
                        if scenario["_label_a"] == "aura":
                            aura_scores.append(float(val_a))
                            base_scores.append(float(val_b))
                        else:
                            aura_scores.append(float(val_b))
                            base_scores.append(float(val_a))

            if len(aura_scores) < 5:
                results[dim] = {"error": "insufficient data", "n": len(aura_scores)}
                continue

            # Manual Wilcoxon signed-rank (no scipy dependency)
            diffs = [a - b for a, b in zip(aura_scores, base_scores) if a != b]
            if not diffs:
                results[dim] = {"W": 0, "p_approx": 1.0, "n": len(aura_scores)}
                continue

            abs_diffs = [(abs(d), d) for d in diffs]
            abs_diffs.sort(key=lambda x: x[0])
            # Assign ranks
            ranks = list(range(1, len(abs_diffs) + 1))
            w_plus = sum(r for r, (_, d) in zip(ranks, abs_diffs) if d > 0)
            w_minus = sum(r for r, (_, d) in zip(ranks, abs_diffs) if d < 0)
            W = min(w_plus, w_minus)
            n = len(diffs)
            # Normal approximation for p-value
            mean_W = n * (n + 1) / 4
            std_W = math.sqrt(n * (n + 1) * (2 * n + 1) / 24) if n > 0 else 1
            z = (W - mean_W) / std_W if std_W > 0 else 0
            # Two-tailed p via normal CDF approximation
            p_approx = 2 * (1 - _norm_cdf(abs(z)))

            results[dim] = {
                "W": W,
                "z": round(z, 4),
                "p_approx": round(p_approx, 4),
                "n_pairs": n,
                "aura_mean": round(sum(aura_scores) / len(aura_scores), 3),
                "baseline_mean": round(sum(base_scores) / len(base_scores), 3),
            }

        return results

    def cohens_kappa(self) -> Optional[float]:
        """Cohen's kappa between the first two annotators (if available)."""
        if len(self.annotations) < 2:
            return None
        a1, a2 = self.annotations[0], self.annotations[1]
        keys = sorted(set(a1.keys()) & set(a2.keys()))
        if not keys:
            return None

        ratings_1 = [int(a1[k]) for k in keys if isinstance(a1.get(k), (int, float))]
        ratings_2 = [int(a2[k]) for k in keys if isinstance(a2.get(k), (int, float))]

        if len(ratings_1) != len(ratings_2) or not ratings_1:
            return None

        n = len(ratings_1)
        categories = sorted(set(ratings_1 + ratings_2))
        # Observed agreement
        po = sum(1 for a, b in zip(ratings_1, ratings_2) if a == b) / n
        # Expected agreement
        pe = sum(
            (ratings_1.count(c) / n) * (ratings_2.count(c) / n) for c in categories
        )
        if pe >= 1.0:
            return 1.0
        return round((po - pe) / (1 - pe), 4)

    def krippendorff_alpha(self) -> Optional[float]:
        """
        Krippendorff's alpha for ordinal data (simplified interval metric).
        """
        if len(self.annotations) < 2:
            return None

        # Collect all items
        items: Dict[str, List[Optional[float]]] = defaultdict(lambda: [None] * len(self.annotations))
        for ai, ann in enumerate(self.annotations):
            for key, val in ann.items():
                if isinstance(val, (int, float)):
                    items[key][ai] = float(val)

        # Filter items with at least 2 ratings
        valid_items = {k: v for k, v in items.items() if sum(1 for x in v if x is not None) >= 2}
        if not valid_items:
            return None

        # Observed disagreement
        Do = 0.0
        n_pairs = 0
        for ratings in valid_items.values():
            vals = [v for v in ratings if v is not None]
            for i in range(len(vals)):
                for j in range(i + 1, len(vals)):
                    Do += (vals[i] - vals[j]) ** 2
                    n_pairs += 1
        if n_pairs == 0:
            return None
        Do /= n_pairs

        # Expected disagreement
        all_values = []
        for ratings in valid_items.values():
            all_values.extend(v for v in ratings if v is not None)
        De = 0.0
        n_total_pairs = 0
        for i in range(len(all_values)):
            for j in range(i + 1, len(all_values)):
                De += (all_values[i] - all_values[j]) ** 2
                n_total_pairs += 1
        if n_total_pairs == 0:
            return None
        De /= n_total_pairs

        if De == 0:
            return 1.0
        return round(1 - Do / De, 4)

    def full_report(self) -> Dict[str, Any]:
        """Generate a complete analysis report."""
        return {
            "means": self.compute_means(),
            "wilcoxon": self.wilcoxon_test(),
            "cohens_kappa": self.cohens_kappa(),
            "krippendorff_alpha": self.krippendorff_alpha(),
            "num_annotators": len(self.annotations),
            "num_scenarios": len(self.scenarios),
        }


def _norm_cdf(x: float) -> float:
    """Standard normal CDF approximation (Abramowitz & Stegun)."""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


# ===========================================================================
# Annotation Collection Server (lightweight Flask-like via http.server)
# ===========================================================================

class AnnotationCollectionServer:
    """
    Minimal HTTP server for collecting human evaluation annotations.
    Serves the HTML form and accepts JSON submissions.

    Usage:
        server = AnnotationCollectionServer(results_dir="evaluation/results")
        server.start(port=5050)  # blocking
    """

    def __init__(self, results_dir: str = "evaluation/results"):
        self.results_dir = Path(results_dir)
        self.annotations_path = self.results_dir / "human_annotations"
        self.annotations_path.mkdir(parents=True, exist_ok=True)

    def start(self, port: int = 5050):
        """Start the annotation collection server."""
        import http.server
        import socketserver

        results_dir = self.results_dir
        annotations_path = self.annotations_path

        class Handler(http.server.SimpleHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/" or self.path == "/eval":
                    html_path = results_dir / "human_eval_forms.html"
                    if html_path.exists():
                        self.send_response(200)
                        self.send_header("Content-Type", "text/html; charset=utf-8")
                        self.end_headers()
                        self.wfile.write(html_path.read_bytes())
                    else:
                        self.send_error(404, "Evaluation form not generated yet")
                elif self.path == "/status":
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    files = list(annotations_path.glob("*.json"))
                    self.wfile.write(json.dumps({
                        "ok": True,
                        "annotations_collected": len(files),
                        "annotators": [f.stem for f in files],
                    }).encode())
                else:
                    super().do_GET()

            def do_POST(self):
                if self.path == "/submit":
                    length = int(self.headers.get("Content-Length", 0))
                    body = self.rfile.read(length)
                    try:
                        data = json.loads(body)
                        annotator_id = data.get("annotator_id", "unknown")
                        is_partial = data.get("partial", False)

                        suffix = "_partial" if is_partial else "_final"
                        out_file = annotations_path / f"{annotator_id}{suffix}.json"
                        with open(out_file, "w", encoding="utf-8") as f:
                            json.dump(data, f, indent=2, ensure_ascii=False)

                        self.send_response(200)
                        self.send_header("Content-Type", "application/json")
                        self.end_headers()
                        self.wfile.write(json.dumps({
                            "ok": True,
                            "saved_to": str(out_file),
                        }).encode())
                    except Exception as e:
                        self.send_response(400)
                        self.send_header("Content-Type", "application/json")
                        self.end_headers()
                        self.wfile.write(json.dumps({
                            "ok": False, "error": str(e)
                        }).encode())
                else:
                    self.send_error(404)

            def log_message(self, format, *args):
                print(f"  [AnnotationServer] {format % args}")

        print(f"Starting annotation collection server on port {port}")
        print(f"  Form URL:   http://localhost:{port}/")
        print(f"  Status URL: http://localhost:{port}/status")
        print(f"  Annotations saved to: {annotations_path}/")

        with socketserver.TCPServer(("", port), Handler) as httpd:
            try:
                httpd.serve_forever()
            except KeyboardInterrupt:
                print("\nAnnotation server stopped.")


# ===========================================================================
# LLM-Judge ↔ Human Agreement Analysis
# ===========================================================================

class JudgeHumanAgreement:
    """
    Analyze agreement between LLM judge scores and human annotations.
    This validates whether the LLM judge is a reliable proxy for human judgment.
    """

    def __init__(
        self,
        scenarios: List[Dict],
        human_annotations: List[Dict],
        llm_results: Dict[str, List[Dict]],
    ):
        self.scenarios = {s["id"]: s for s in scenarios}
        self.human_annotations = human_annotations
        self.llm_results = llm_results  # condition -> list of per-query results

    def compute_agreement(self) -> Dict[str, Any]:
        """Compute correlation and agreement between LLM judge and human ratings."""
        # Map scenario_id -> LLM factual accuracy score
        llm_scores_by_id: Dict[int, Dict[str, float]] = {}
        for cond, results in self.llm_results.items():
            for r in results:
                qid = r.get("query_id")
                if qid is not None and "factual_accuracy" in r:
                    fa = r["factual_accuracy"]
                    if isinstance(fa, dict) and "accuracy" in fa:
                        llm_scores_by_id.setdefault(qid, {})[cond] = fa["accuracy"]

        # Collect paired (human, llm) scores
        pairs: Dict[str, List[Tuple[float, float]]] = defaultdict(list)

        for ann in self.human_annotations:
            ratings = ann.get("ratings", ann)
            for sid, scenario in self.scenarios.items():
                for dim in EVAL_DIMENSIONS:
                    key_a = f"s{sid}_a_{dim}"
                    key_b = f"s{sid}_b_{dim}"
                    human_a = ratings.get(key_a)
                    human_b = ratings.get(key_b)

                    # Normalize human 1-5 to 0-1
                    if human_a is not None:
                        human_norm = (float(human_a) - 1) / 4.0
                        # Find matching LLM score
                        label = scenario.get("_label_a", "")
                        llm_score = llm_scores_by_id.get(sid, {}).get(
                            "AURA_Full" if label == "aura" else "Vanilla_LLM", None
                        )
                        if llm_score is not None:
                            pairs[dim].append((human_norm, llm_score))

                    if human_b is not None:
                        human_norm = (float(human_b) - 1) / 4.0
                        label = scenario.get("_label_b", "")
                        llm_score = llm_scores_by_id.get(sid, {}).get(
                            "AURA_Full" if label == "aura" else "Vanilla_LLM", None
                        )
                        if llm_score is not None:
                            pairs[dim].append((human_norm, llm_score))

        # Compute Pearson correlation per dimension
        agreement = {}
        for dim, pair_list in pairs.items():
            if len(pair_list) < 5:
                agreement[dim] = {"error": "insufficient data", "n": len(pair_list)}
                continue

            human_vals = [p[0] for p in pair_list]
            llm_vals = [p[1] for p in pair_list]

            # Pearson r
            n = len(pair_list)
            h_mean = sum(human_vals) / n
            l_mean = sum(llm_vals) / n
            cov = sum((h - h_mean) * (l - l_mean) for h, l in zip(human_vals, llm_vals)) / n
            h_std = math.sqrt(sum((h - h_mean) ** 2 for h in human_vals) / n)
            l_std = math.sqrt(sum((l - l_mean) ** 2 for l in llm_vals) / n)
            r = cov / (h_std * l_std) if h_std > 0 and l_std > 0 else 0

            # Mean absolute error
            mae = sum(abs(h - l) for h, l in zip(human_vals, llm_vals)) / n

            agreement[dim] = {
                "pearson_r": round(r, 4),
                "mae": round(mae, 4),
                "n_pairs": n,
                "human_mean": round(h_mean, 4),
                "llm_mean": round(l_mean, 4),
                "interpretation": (
                    "strong agreement" if r > 0.7
                    else "moderate agreement" if r > 0.4
                    else "weak agreement" if r > 0.2
                    else "no agreement"
                ),
            }

        return {
            "per_dimension": agreement,
            "overall_interpretation": (
                "LLM judge is a reliable proxy for human judgment"
                if all(
                    a.get("pearson_r", 0) > 0.4
                    for a in agreement.values()
                    if isinstance(a, dict) and "pearson_r" in a
                )
                else "LLM judge shows mixed agreement with human judgment — "
                     "interpret LLM-only results with caution"
            ),
        }


# ===========================================================================
# Run RQ5 analysis on collected annotations
# ===========================================================================

def analyze_human_eval(results_dir: str = "evaluation/results") -> Optional[Dict[str, Any]]:
    """
    Analyze collected human annotations.
    Call this after annotators have submitted their ratings.
    """
    results_path = Path(results_dir)

    # Load scenarios
    forms_path = results_path / "human_eval_forms.json"
    if not forms_path.exists():
        print("  [SKIP] Human eval forms not generated yet")
        return None

    with open(forms_path) as f:
        scenarios = json.load(f)

    # Load annotations
    ann_dir = results_path / "human_annotations"
    if not ann_dir.exists():
        print("  [SKIP] No annotations directory found")
        return None

    annotations = []
    for ann_file in sorted(ann_dir.glob("*_final.json")):
        with open(ann_file) as f:
            ann_data = json.load(f)
        annotations.append(ann_data.get("ratings", ann_data))

    if not annotations:
        print("  [SKIP] No final annotations found")
        return None

    print(f"  Loaded {len(annotations)} annotator(s), {len(scenarios)} scenarios")

    # Run analysis
    analyzer = HumanEvalAnalyzer(scenarios, annotations)
    report = analyzer.full_report()

    # Save
    out_path = results_path / "rq5_human_eval_analysis.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"  -> Analysis saved to {out_path}")

    # Print summary
    print("\n  === Human Evaluation Results ===")
    means = report.get("means", {})
    for system in ["aura", "baseline"]:
        scores = means.get(system, {})
        print(f"  {system}: " + ", ".join(f"{d}={s:.2f}" for d, s in scores.items()))

    wilcoxon = report.get("wilcoxon", {})
    for dim, result in wilcoxon.items():
        if isinstance(result, dict) and "p_approx" in result:
            sig = "*" if result["p_approx"] < 0.05 else "n.s."
            print(f"  Wilcoxon {dim}: p={result['p_approx']:.4f} {sig}")

    alpha = report.get("krippendorff_alpha")
    kappa = report.get("cohens_kappa")
    if alpha is not None:
        print(f"  Krippendorff's alpha: {alpha:.4f}")
    if kappa is not None:
        print(f"  Cohen's kappa: {kappa:.4f}")

    return report
