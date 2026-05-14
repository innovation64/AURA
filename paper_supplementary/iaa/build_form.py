"""Generate the bilingual (EN/ZH) IAA annotation HTML form for the 25
implicit-intent queries.

Two annotation tasks per query:
  Task A: classify the implicit need into one of 5 subcategories
          (availability / mood / appropriateness / latent_goal / second_order)
          or "literal" if the surface query needs no implicit-need inference.
  Task B: 1-line free-text "what is the user really asking?"

We use Task A's categorical answer for Cohen's κ (the rigorous IAA stat).
Task B is logged but not auto-scored — gives readers qualitative evidence.

Output: evaluation/iaa_implicit_intent/iaa_form.html
        Annotators open it in a browser, fill in, click "Download my answers".
        Browser downloads `iaa_<your_name>.json`.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
QUERIES = ROOT / "evaluation" / "data" / "implicit_intent_queries.json"

SCENE_EN = (
    "AURATown is a small simulated town with 5 residents going about their day. "
    "Lin Wei runs Sunrise Cafe; Zhang Hao is a writer who often works at the cafe; "
    "Chen Mei runs the General Store; Liu Yang is a college student; "
    "Wang Jun is a retired professor who likes the library. Each agent has a public "
    "state (location, what they're doing) AND a private state (whether they are "
    "available, their mood, unspoken goals, what they currently believe about "
    "the other residents). The private state is NOT visible — it has to be looked up."
)

SCENE_ZH = (
    "AURATown 是一个有 5 位居民的小镇模拟。"
    "林伟（Lin Wei）经营 Sunrise Cafe；张浩（Zhang Hao）是一位常在咖啡馆写作的作家；"
    "陈梅（Chen Mei）经营 General Store；刘洋（Liu Yang）是大学生；"
    "王军（Wang Jun）是退休教授，喜欢去图书馆。每位居民都有"
    "公开状态（位置、正在做什么）和私有状态（当前是否有空、情绪、"
    "未说出的目标、对其他居民的当下看法）。私有状态默认不可见，"
    "需要专门去查询。"
)

CATS = [
    (
        "availability",
        "the user really wants to know if the named person is FREE to be approached / interrupted",
        "用户其实想问：这个人现在是否有空 / 能不能被打扰",
    ),
    (
        "mood",
        "the user really wants to know how the person is FEELING (emotionally)",
        "用户其实想问：这个人现在的情绪状态如何",
    ),
    (
        "appropriateness",
        "the user wants to know if NOW IS A GOOD TIME for some action (e.g., to invite, to interrupt)",
        "用户想问：现在是不是做某件事的好时机（如邀请、打扰、提议）",
    ),
    (
        "latent_goal",
        "the user wants to know what the person is really TRYING TO DO (their unspoken goal)",
        "用户想问：这个人真正在做什么 / 想达成什么（未说出的目标）",
    ),
    (
        "second_order",
        "the user wants to know what one person BELIEVES ABOUT another (what X thinks about Y)",
        "用户想问：一个人对另一个人的看法是什么（X 是怎么看 Y 的）",
    ),
    (
        "literal",
        "the surface query is the actual need; no hidden private-state need behind it",
        "字面问题就是真实需求，背后没有隐藏的私有状态需求",
    ),
]

# Authoritative Chinese translations of the 25 queries. Keyed by query id.
# Names kept in pinyin (matches scene description); style kept colloquial to
# match the English source.
QUERY_ZH = {
    101: "林伟在哪？",
    102: "张浩在做什么？",
    103: "陈梅在社区菜园吗？",
    104: "王军还在图书馆吗？",
    105: "现在 Sunrise Cafe 有谁？",
    106: "林伟今天怎么样？",
    107: "张浩看起来忙吗？",
    108: "咖啡馆现在的氛围如何？",
    109: "陈梅今天过得好吗？",
    110: "今天有谁看起来压力很大吗？",
    111: "现在邀请林伟喝杯咖啡合适吗？",
    112: "我今天该顺路去一下张浩家吗？",
    113: "现在跟咖啡馆里这群人提问合适吗？",
    114: "我现在能找王军帮个忙吗？",
    115: "我今天该跟陈梅提那个种植计划吗？",
    116: "林伟最近在忙什么？",
    117: "张浩为什么去图书馆？",
    118: "陈梅这个下午有什么打算？",
    119: "今天有谁在躲着谁吗？",
    120: "王军最近在想什么？",
    121: "林伟觉得现在张浩可以被打扰吗？",
    122: "从张浩的角度看，陈梅现在有心情聊天吗？",
    123: "陈梅是不是觉得林伟现在太忙没法聊？",
    124: "王军觉得林伟今天心情怎么样？",
    125: "林伟是不是认为陈梅现在想一个人待着？",
}


def build() -> str:
    with open(QUERIES) as f:
        data = json.load(f)
    queries = data.get("queries", data if isinstance(data, list) else [])

    # Sanity check: every query has a Chinese translation
    missing_zh = [q["id"] for q in queries if q["id"] not in QUERY_ZH]
    if missing_zh:
        raise RuntimeError(
            f"missing Chinese translation for query ids: {missing_zh}. "
            f"add them to QUERY_ZH in build_form.py."
        )

    # Randomise display order so annotators don't see subcategory groupings
    rng = random.Random(20260429)
    display_order = list(range(len(queries)))
    rng.shuffle(display_order)

    cats_html = "\n".join(
        f'<div class="cat"><b>{cat}</b>: <span class="en">{desc_en}</span>'
        f'<br><span class="zh">{desc_zh}</span></div>'
        for cat, desc_en, desc_zh in CATS
    )

    items_html = []
    for display_pos, idx in enumerate(display_order):
        q = queries[idx]
        qid = q["id"]
        text_en = q["query"]
        text_zh = QUERY_ZH[qid]
        agent = q.get("agent_subject", "") or "(no specific person / 不指向具体的人)"

        radios = "\n".join(
            f'<label><input type="radio" name="q{qid}_cat" value="{cat}" required> '
            f'<b>{cat}</b></label>'
            for cat, _, _ in CATS
        )
        items_html.append(f"""
<div class="item" id="item-{qid}">
  <div class="qhead">Q{display_pos + 1} (id={qid})</div>
  <div class="qtext en">EN: "{text_en}" <span class="who">— asked about <b>{agent}</b></span></div>
  <div class="qtext zh">中: 「{text_zh}」 <span class="who">— 关于 <b>{agent}</b></span></div>
  <div class="prompt"><b>A.</b> Which of the 6 categories best describes what the user is <em>really</em> asking?<br>
  <span class="zh">用户真正在问的内容，最符合下面 6 个类别中的哪一个？</span></div>
  <div class="radios">{radios}</div>
  <div class="prompt"><b>B.</b> (optional) In one sentence: what is the user really asking?<br>
  <span class="zh">（可选）用一句话写下你认为用户真正在问什么。中英任一即可。</span></div>
  <textarea name="q{qid}_text" rows="2" placeholder="e.g. 'Whether Lin Wei is free to chat right now' / 例如：「林伟现在是不是有空聊」"></textarea>
</div>
""")

    items_block = "\n".join(items_html)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>AURA Implicit-Intent IAA Form / 隐式意图标注表</title>
<style>
  body {{ font-family: system-ui, -apple-system, "PingFang SC", "Microsoft YaHei", sans-serif; max-width: 820px; margin: 24px auto; padding: 0 18px; color: #1f2937; }}
  h1 {{ color: #0f172a; margin-bottom: 4px; }}
  h1 .subtitle {{ font-size: 0.55em; font-weight: 400; color: #64748b; display: block; margin-top: 2px; }}
  h2 {{ color: #0f172a; margin-top: 28px; }}
  h2 .zh {{ color: #64748b; font-weight: 400; font-size: 0.7em; margin-left: 6px; }}
  .scene {{ background: #f1f5f9; padding: 12px 16px; border-radius: 8px; font-size: 0.95em; line-height: 1.6; }}
  .scene .zh {{ display: block; margin-top: 8px; padding-top: 8px; border-top: 1px dashed #cbd5e1; color: #334155; }}
  .cats {{ background: #fef3c7; padding: 12px 16px; border-radius: 8px; margin-top: 12px; font-size: 0.92em; line-height: 1.5; }}
  .cat {{ margin: 6px 0; }}
  .cat .zh {{ color: #475569; font-size: 0.95em; }}
  .item {{ border: 1px solid #e5e7eb; border-radius: 8px; padding: 12px 16px; margin: 14px 0; background: #ffffff; }}
  .qhead {{ font-weight: 700; color: #0f172a; font-size: 0.9em; }}
  .qtext {{ margin: 6px 0; font-size: 1.05em; }}
  .qtext.zh {{ font-size: 0.98em; color: #334155; }}
  .who {{ color: #64748b; font-size: 0.88em; }}
  .prompt {{ font-size: 0.9em; color: #475569; margin-top: 10px; line-height: 1.45; }}
  .prompt .zh {{ color: #64748b; font-size: 0.92em; }}
  .radios label {{ display: block; margin: 4px 0; cursor: pointer; }}
  textarea {{ width: 100%; box-sizing: border-box; padding: 6px 8px; font-family: inherit; font-size: 0.95em; }}
  #name {{ font-size: 1em; padding: 6px 10px; width: 280px; }}
  button {{ background: #16a34a; color: white; border: none; padding: 10px 18px; border-radius: 6px; font-size: 1em; cursor: pointer; margin-top: 18px; }}
  button:hover {{ background: #15803d; }}
  .progress {{ position: sticky; top: 0; background: #fff; padding: 10px 0; border-bottom: 1px solid #e5e7eb; z-index: 10; }}
  .lang-toggle {{ float: right; font-size: 0.85em; color: #475569; background: #f1f5f9; padding: 6px 10px; border-radius: 6px; cursor: pointer; user-select: none; }}
  body.zh-only .en {{ display: none; }}
  body.en-only .zh {{ display: none; }}
</style>
</head>
<body>

<div class="lang-toggle" onclick="cycleLang()" id="langtoggle">View: EN + 中文 (click to toggle)</div>

<h1>AURA Implicit-Intent IAA Form
<span class="subtitle">隐式意图标注一致性检验表（25 题）</span></h1>

<p class="en">You are about to read 25 short user queries. For each, decide which of 6 categories best
describes what the user is <em>really</em> asking. This takes ~15-20 minutes.</p>
<p class="zh">下面有 25 条用户提问。对每一条，从 6 个类别中选出最能描述
用户<em>真正</em>意图的那一个。整个过程约 15–20 分钟。</p>

<h2>Setting <span class="zh">场景设定</span></h2>
<div class="scene"><span class="en">{SCENE_EN}</span><span class="zh">{SCENE_ZH}</span></div>

<h2>Categories <span class="zh">类别</span></h2>
<div class="cats">{cats_html}</div>

<h2>Your name <span class="zh">你的名字</span></h2>
<p class="en">Used only to label your saved file. Anything is fine — initials, a nickname.</p>
<p class="zh">只用于命名导出的文件。随便填，例如缩写、昵称、R1 都可以。</p>
<input id="name" placeholder="e.g. R1, Alex, dh, …" required>

<h2>The 25 questions <span class="zh">25 题</span></h2>
<form id="iaaform">
  <div class="progress" id="progress">0 / 25 answered · 已答 0 / 25</div>
  {items_block}
  <button type="button" onclick="submitForm()">Download my answers / 下载我的标注</button>
</form>

<script>
function submitForm() {{
  const name = document.getElementById('name').value.trim() || 'anon';
  const items = document.querySelectorAll('.item');
  const out = {{ annotator: name, n_questions: items.length, ratings: {{}}, free_text: {{}} }};
  let answered = 0;
  let missing = [];
  items.forEach(it => {{
    const qid = it.id.replace('item-','');
    const radio = it.querySelector('input[type="radio"]:checked');
    const text = it.querySelector('textarea').value.trim();
    if (radio) {{
      out.ratings[qid] = radio.value;
      answered++;
    }} else {{
      missing.push(qid);
    }}
    if (text) out.free_text[qid] = text;
  }});
  if (missing.length > 0) {{
    alert('Missing answer for question id(s) / 以下题号还没标注: ' + missing.join(', ') + '. Please complete or scroll up. / 请补齐后再下载。');
    return;
  }}
  const blob = new Blob([JSON.stringify(out, null, 2)], {{type: 'application/json'}});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'iaa_' + name.replace(/[^a-zA-Z0-9_-]/g, '_') + '.json';
  a.click();
  URL.revokeObjectURL(url);
}}

// Live progress
function updateProgress() {{
  const items = document.querySelectorAll('.item');
  let answered = 0;
  items.forEach(it => {{ if (it.querySelector('input[type="radio"]:checked')) answered++; }});
  document.getElementById('progress').textContent =
    answered + ' / ' + items.length + ' answered · 已答 ' + answered + ' / ' + items.length;
}}
document.addEventListener('change', updateProgress);

// Language toggle: EN+ZH (default) → EN only → ZH only → EN+ZH
const LANG_STATES = ['both', 'en', 'zh'];
const LANG_LABELS = {{
  both: 'View: EN + 中文 (click to toggle)',
  en:   'View: EN only (click for 中文)',
  zh:   '视图: 仅中文 (点击切换为 EN+中文)',
}};
let langIdx = 0;
function cycleLang() {{
  langIdx = (langIdx + 1) % LANG_STATES.length;
  const state = LANG_STATES[langIdx];
  document.body.classList.remove('zh-only', 'en-only');
  if (state === 'en') document.body.classList.add('zh-only'); // hide .zh, show .en
  if (state === 'zh') document.body.classList.add('en-only'); // hide .en, show .zh
  document.getElementById('langtoggle').textContent = LANG_LABELS[state];
}}
</script>
</body>
</html>
"""
    return html


def main() -> int:
    out_path = Path(__file__).parent / "iaa_form.html"
    out_path.write_text(build(), encoding="utf-8")
    print(f"wrote {out_path}")
    print(f"  → open it in a browser, fill in 25 questions, click Download")
    print(f"  → save returned iaa_<name>.json into evaluation/iaa_implicit_intent/responses/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
