#!/usr/bin/env python3
"""Build the AURA-Intent vs NoIntent human-evaluation arm (camera-ready commitment to Reviewer vBZF).

Pairs are joined from already-logged per-query answers in rq_intent_v2_multiseed.json
(seed 42, conditions `tom` vs `no_intent`), so no LLM calls are needed. Produces:
  evaluation/results_nointent/human_eval_forms.json   (same schema as the RQ5 file; labels aura/baseline)
  evaluation/results_nointent/human_eval_offline.html (self-contained rater form, downloads ratings JSON)
Rating JSON shape matches evaluation/results/annotations/*.json so compute_irr.py / rq5 aggregation reuse.
"""
import json, random, re, html, argparse
from pathlib import Path

DIMS = [
    ("response_helpfulness", "Response helpfulness", "Does the answer actually help the user with what they need (including what they did not literally ask)?"),
    ("environmental_awareness", "Environmental awareness", "Does the answer reflect awareness of the situation the target agent is in (availability, mood, plans)?"),
    ("agent_believability", "Agent believability", "Does the response read like a believable, situated assistant rather than a generic chatbot?"),
    ("factual_accuracy", "Factual accuracy", "Check against Scene facts: does the answer avoid inventing details, AND avoid asserting things that contradict the scene (e.g. saying nothing is planned when the facts show something is)?"),
]
HEADS_UP = re.compile(r"^\s*\[heads-up\][^\n]*\n?", re.IGNORECASE)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="evaluation/results/rq_intent_v2_multiseed.json")
    ap.add_argument("--bench", default="evaluation/data/implicit_intent_queries_v2.json")
    ap.add_argument("--seed", default="42")
    ap.add_argument("--n-per-subcat", type=int, default=10)
    ap.add_argument("--out-dir", default="evaluation/results_nointent")
    ap.add_argument("--keep-heads-up", action="store_true", help="keep the [heads-up] prefix (breaks blinding)")
    args = ap.parse_args()

    d = json.load(open(args.src))
    bench = json.load(open(args.bench))
    scenes = bench["scenes"]
    tom = {r["query_id"]: r for r in d["per_seed"][args.seed]["tom"]}
    noi = {r["query_id"]: r for r in d["per_seed"][args.seed]["no_intent"]}
    shared = sorted(set(tom) & set(noi))
    rng = random.Random(42)
    by_sub = {}
    for q in shared:
        by_sub.setdefault(tom[q]["subcategory"], []).append(q)
    chosen = []
    for sub in sorted(by_sub):
        qs = by_sub[sub][:]
        rng.shuffle(qs)
        chosen += qs[: args.n_per_subcat]
    rng.shuffle(chosen)

    items = []
    for i, q in enumerate(chosen):
        a = tom[q]["answer"] or ""
        if not args.keep_heads_up:
            a = HEADS_UP.sub("", a).strip()
        b = noi[q]["answer"] or ""
        flip = rng.random() < 0.5
        items.append({
            "id": i,
            "query": tom[q]["query"],
            "agent": tom[q].get("agent_subject", ""),
            "category": tom[q]["subcategory"],
            "response_a": b if flip else a,
            "response_b": a if flip else b,
            "_label_a": "baseline" if flip else "aura",
            "_label_b": "aura" if flip else "baseline",
            "_query_id": q,
            "_scene": tom[q]["scene"],
            "_scene_facts": _facts(scenes[tom[q]["scene"]], tom[q]["subcategory"]),
            "_arm": "intent_vs_nointent",
        })

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "human_eval_forms.json").write_text(json.dumps(items, indent=2, ensure_ascii=False))
    (out / "human_eval_offline.html").write_text(render_html(items))
    print(f"wrote {len(items)} items -> {out}/human_eval_forms.json and human_eval_offline.html")
    from collections import Counter
    print("per-subcategory:", dict(Counter(x["category"] for x in items)))
    print("A=aura count:", sum(x["_label_a"] == "aura" for x in items))

def _facts(scene, subcat):
    """Ground-truth state a rater needs to score factual accuracy.

    Includes public + private state for every agent. Deliberately EXCLUDES the
    benchmark's `implicit_need` annotation, which is the answer key for what a
    good response should surface -- showing it would tell raters which response
    to prefer. Beliefs-about-others are shown only for second_order items, where
    the question is about a belief rather than a fact.
    """
    out = {"summary": scene.get("summary", ""), "agents": {}}
    for name, pub in scene.get("public_state", {}).items():
        priv = scene.get("private_state", {}).get(name, {})
        out["agents"][name] = {
            "location": pub.get("location", ""),
            "doing (visible)": pub.get("action", ""),
            "availability (hidden)": priv.get("availability", ""),
            "mood (hidden)": priv.get("emotional_state", ""),
            "unspoken goal (hidden)": priv.get("unspoken_goal", ""),
        }
    if subcat == "second_order":
        out["beliefs_about_others"] = scene.get("beliefs_about_others", {})
    return out


def _facts_html(it):
    esc = html.escape
    f = it["_scene_facts"]
    rows = ""
    for name, a in f["agents"].items():
        cells = "".join(f"<td>{esc(str(v))}</td>" for v in a.values())
        rows += f"<tr><td><b>{esc(name)}</b></td>{cells}</tr>"
    beliefs = ""
    if "beliefs_about_others" in f:
        beliefs = "<p><b>What each agent believes about the others:</b> " + esc(json.dumps(f["beliefs_about_others"], ensure_ascii=False)) + "</p>"
    return f'''<details class="facts"><summary>Scene facts &mdash; open this to score <b>factual accuracy</b></summary>
    <p class="ctx">{esc(f["summary"])}</p>
    <table class="factbl"><thead><tr><th>Agent</th><th>Location</th><th>Doing (visible)</th><th>Availability (hidden)</th><th>Mood (hidden)</th><th>Unspoken goal (hidden)</th></tr></thead><tbody>{rows}</tbody></table>
    {beliefs}
    <p class="note">Hidden columns are true of the world but not visible in the scene &mdash; a system must probe for them. Use this table only to check whether a response <em>contradicts</em> these facts or <em>invents</em> details, and whether it missed something the question asked about. It does not tell you which response is "correct".</p>
    </details>'''


def render_html(items):
    esc = html.escape
    cards = []
    for it in items:
        rows = ""
        for key, label, hint in DIMS:
            def radios(side):
                return "".join(
                    f'<label><input type="radio" name="s{it["id"]}_{side}_{key}" value="{v}" required> {v}</label>'
                    for v in range(1, 6))
            rows += f'<tr><td class="dim"><b>{esc(label)}</b><br><small>{esc(hint)}</small></td><td>{radios("a")}</td><td>{radios("b")}</td></tr>'
        cards.append(f'''
<section class="card" data-sid="{it["id"]}">
  <h3>Scenario {it["id"]+1} / {len(items)}</h3>
  <p><b>User asks about agent:</b> {esc(it["agent"])}</p>
  <p class="q"><b>Query:</b> {esc(it["query"])}</p>
  {_facts_html(it)}
  <div class="pair">
    <div class="resp"><h4>Response A</h4><pre>{esc(it["response_a"])}</pre></div>
    <div class="resp"><h4>Response B</h4><pre>{esc(it["response_b"])}</pre></div>
  </div>
  <table><thead><tr><th>Dimension (1 = poor, 5 = excellent)</th><th>Response A</th><th>Response B</th></tr></thead><tbody>{rows}</tbody></table>
</section>''')
    body = "\n".join(cards)
    return f'''<!doctype html><html><head><meta charset="utf-8"><title>AURA human evaluation — Intent vs NoIntent arm</title>
<style>
body{{font-family:system-ui,sans-serif;max-width:1100px;margin:2rem auto;padding:0 1rem;line-height:1.4}}
.card{{border:1px solid #ccc;border-radius:8px;padding:1rem;margin:1.5rem 0}}
.pair{{display:grid;grid-template-columns:1fr 1fr;gap:1rem}}
.resp pre{{white-space:pre-wrap;background:#f6f6f6;padding:.75rem;border-radius:6px;font-family:inherit}}
.facts{{background:#fffbe6;border:1px solid #e6d48a;border-radius:6px;padding:.5rem .75rem;margin:.6rem 0}}
.facts summary{{cursor:pointer;font-size:.95rem}}
.factbl{{font-size:.85rem}} .factbl td,.factbl th{{padding:.25rem .4rem}}
.note{{font-size:.8rem;color:#555}} .ctx{{font-size:.9rem;font-style:italic}}
table{{width:100%;border-collapse:collapse;margin-top:.5rem}} td,th{{border-top:1px solid #ddd;padding:.4rem;vertical-align:top}}
td label{{margin-right:.6rem;white-space:nowrap}} .dim{{width:45%}}
#top{{position:sticky;top:0;background:#fff;padding:.5rem 0;border-bottom:1px solid #ccc}}
button{{padding:.5rem 1rem;font-size:1rem}}
</style></head><body>
<div id="top">
 <b>AURA human evaluation — arm 2 (Intent vs NoIntent)</b> &nbsp; Rater ID: <input id="rid" placeholder="your initials" size="10">
 &nbsp; <span id="progress">0 / {len(items)} complete</span> &nbsp; <button onclick="download()">Download final ratings</button>
 <p style="margin:.3rem 0 0;font-size:.9rem">You will see {len(items)} situated queries about agents in a small simulated town, each with two anonymous responses. Rate BOTH responses on each of four dimensions (1–5). The two responses come from two different systems in random order. Each scenario has a collapsible <b>Scene facts</b> box: open it before scoring <b>factual accuracy</b>, since you cannot judge that from the two responses alone. Answers autosave in this browser; press Download when finished and send the file back.</p>
</div>
{body}
<script>
const KEY='aura_nointent_eval';
const inputs=[...document.querySelectorAll('input[type=radio]')];
const total={len(items)};
function state(){{const s={{}};inputs.forEach(i=>{{if(i.checked)s[i.name]=+i.value}});return s;}}
function save(){{try{{localStorage.setItem(KEY,JSON.stringify({{rid:document.getElementById('rid').value,ratings:state()}}))}}catch(e){{}};prog();}}
function prog(){{const s=state();let done=0;for(let k=0;k<total;k++){{let ok=true;for(const d of {json.dumps([d[0] for d in DIMS])}){{if(!(`s${{k}}_a_${{d}}` in s)||!(`s${{k}}_b_${{d}}` in s))ok=false;}}if(ok)done++;}}document.getElementById('progress').textContent=done+' / '+total+' complete';return done;}}
function download(){{const done=prog();const rid=document.getElementById('rid').value||'anon';
 const payload={{annotator_id:rid,ratings:state(),completed_scenarios:done,total_scenarios:total,is_complete:done===total,rater_kind:'human_offline',arm:'intent_vs_nointent',submitted_at:new Date().toISOString()}};
 const a=document.createElement('a');a.href='data:application/json;charset=utf-8,'+encodeURIComponent(JSON.stringify(payload,null,2));a.download='nointent_ratings_'+rid+'.json';a.click();}}
inputs.forEach(i=>i.addEventListener('change',save));document.getElementById('rid').addEventListener('input',save);
try{{const st=JSON.parse(localStorage.getItem(KEY)||'null');if(st){{document.getElementById('rid').value=st.rid||'';for(const [k,v] of Object.entries(st.ratings||{{}})){{const el=document.querySelector(`input[name="${{k}}"][value="${{v}}"]`);if(el)el.checked=true;}}}}}}catch(e){{}}
prog();
</script></body></html>'''

if __name__ == "__main__":
    main()
