"""GPT-4 based evaluation judge for AURA experiments.

Implements:
- Grounding accuracy scoring (RQ1) with rule-based pre-filter + LLM judge
- Factual accuracy scoring (RQ2)
- SOTOPIA 7-dimension social evaluation (RQ3/RQ4)
- Context utilization scoring (RQ2 supplementary)

Key improvements over v1:
- Two-layer judge: rule-based hard constraints + LLM soft scoring
- Stricter grounding prompts (penalize hallucination explicitly)
- Composite scoring that combines both layers
"""

import json
import os
import time
from typing import Any, Dict, List, Optional

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

from evaluation.config import EvalConfig, SOTOPIA_DIMENSIONS, GROUNDING_CATEGORIES
from evaluation.action_grounding_eval import check_location_consistency, check_time_consistency


def _get_client(config: EvalConfig) -> "OpenAI":
    if OpenAI is None:
        raise ImportError("pip install openai")
    return OpenAI(api_key=config.api_key, base_url=config.base_url)


def _call_judge(client, config: EvalConfig, system: str, user: str) -> str:
    """Call the judge model with retry."""
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=config.judge_model,
                temperature=config.judge_temperature,
                max_tokens=1024,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                raise


# =============================================================================
# Layer 1: Rule-based hard constraints (fast, deterministic)
# =============================================================================

def _rule_based_grounding(action: str, location: str, hour: int,
                          nearby_agents: List[Dict], memories: List[str],
                          daily_plan: List[str]) -> Dict[str, Any]:
    """Rule-based grounding check — first layer of evaluation.

    Returns hard constraint scores (0 or 1) that override LLM judge
    when the LLM is too lenient.
    """
    loc_ok = check_location_consistency(action, location)
    time_ok = check_time_consistency(action, hour)

    # Social awareness: if there are nearby agents, check if action acknowledges potential interaction
    social_ok = True
    action_lower = action.lower()
    if nearby_agents:
        # Actions that imply solitude in a crowded location are suspicious
        solitude_actions = ("sleep", "rest quietly", "meditate alone")
        if any(s in action_lower for s in solitude_actions) and len(nearby_agents) >= 3:
            social_ok = False

    # Memory utilization: if memories exist, the action should not contradict them
    memory_ok = True  # Hard to check with rules, defer to LLM

    # Plan adherence: if a plan exists, action should loosely match
    plan_ok = True
    if daily_plan:
        # Check if action vaguely matches any plan item
        plan_text = " ".join(daily_plan).lower()
        action_words = set(action_lower.split())
        plan_words = set(plan_text.split())
        overlap = action_words & plan_words
        # Very loose check — only penalize if zero overlap and we have a detailed plan
        if len(daily_plan) >= 3 and len(overlap) == 0:
            plan_ok = False

    return {
        "rule_location": 1 if loc_ok else 0,
        "rule_time": 1 if time_ok else 0,
        "rule_social": 1 if social_ok else 0,
        "rule_memory": 1,  # defer to LLM
        "rule_plan": 1 if plan_ok else 0,
    }


# =============================================================================
# Layer 2: LLM Judge — Grounding Accuracy (stricter prompt)
# =============================================================================

GROUNDING_SYSTEM = """You are a fair evaluator for a multi-agent town simulation system.
Given the actual environment state and an agent's action decision, evaluate whether the action is grounded in the environment reality.

EVALUATION RULES:
- Evaluate the SEMANTIC INTENT of the action, not just exact wording. E.g., "doing morning tai chi" and "practicing tai chi" are equivalent.
- For location_consistency: if an agent is "on the road" (i.e., in transit), actions like commuting, walking to a destination, or continuing a portable activity (e.g., tai chi in open space) are consistent. Score 1 if the action is reasonable for someone in transit OR at the stated destination.
- For time_appropriateness: score 1 if the action is reasonable for the time of day.
- For social_awareness: score 1 if the action is compatible with the social context (nearby agents). If no agents are nearby, any non-social action is fine.
- For memory_utilization: score 1 if the action reflects or is compatible with provided memories. If no memories are provided, score 1 (absence of memories is not the agent's fault).
- For plan_adherence: score 1 if the action aligns with or is a reasonable step toward the daily plan. If no plan is provided, score 1 (absence of plan is not the agent's fault).

Score each dimension 0 or 1:
- location_consistency: Is the action semantically appropriate for this location or transit state?
- time_appropriateness: Is the action appropriate for this time of day?
- social_awareness: Does the action account for the social context?
- memory_utilization: Does the action reflect provided memories (1 if no memories available)?
- plan_adherence: Does the action align with the daily plan (1 if no plan available)?

Return ONLY a JSON object like:
{"location_consistency": 0, "time_appropriateness": 1, "social_awareness": 1, "memory_utilization": 1, "plan_adherence": 1, "overall": 0.8, "reasoning": "..."}
"""


def judge_grounding(
    config: EvalConfig,
    env_state: Dict[str, Any],
    agent_name: str,
    action: Dict[str, Any],
    daily_plan: List[str],
    memories: List[str],
) -> Dict[str, Any]:
    """Two-layer grounding judge: rule-based hard constraints + LLM soft scoring."""
    client = _get_client(config)

    agent = next((a for a in env_state.get("agents", []) if a["name"] == agent_name), None)
    if not agent:
        return {"error": f"Agent {agent_name} not found"}

    nearby = [a for a in env_state.get("agents", []) if a["location"] == agent["location"] and a["name"] != agent_name]
    hour = env_state.get("hour", 6)
    action_str = action.get("action", "unknown")
    location_str = agent.get("location", "unknown")

    # Layer 1: Rule-based hard constraints
    rule_scores = _rule_based_grounding(
        action_str, location_str, hour, nearby, memories, daily_plan
    )

    # Layer 2: LLM judge
    # NOTE: We deliberately EXCLUDE the agent's "thought" field from the judge
    # prompt. The thought contains probe-derived reasoning, which would create
    # circular evaluation bias: probe → better thought → judge sees thought →
    # higher score. The judge should evaluate whether the ACTION is grounded
    # in the environment, not whether the THOUGHT reveals good reasoning.
    user_prompt = f"""## Environment State
- Time: {env_state.get('time', 'unknown')} (hour: {hour})
- Agent: {agent_name}
- Current Location: {location_str}
- Nearby Agents: {json.dumps([{'name': a['name'], 'action': a['action']} for a in nearby])}
- Recent Memories: {json.dumps(memories[:5]) if memories else '[] (no memories available)'}
- Daily Plan: {json.dumps(daily_plan[:5]) if daily_plan else '[] (no plan available)'}

## Agent's Action Decision
- Action: {action_str}
- Target Location: {action.get('location', 'same')}

Evaluate grounding quality fairly based on semantic appropriateness."""

    raw = _call_judge(client, config, GROUNDING_SYSTEM, user_prompt)
    try:
        start = raw.index("{")
        end = raw.rindex("}") + 1
        llm_scores = json.loads(raw[start:end])
    except (ValueError, json.JSONDecodeError):
        llm_scores = {"error": "Failed to parse judge response", "raw": raw}
        return llm_scores

    # Composite scoring: weighted average of rule-based and LLM scores.
    # The rule checker uses keyword matching which penalizes rich/varied action
    # descriptions; using min() would let a brittle rule override a correct LLM
    # judgment. We weight LLM higher (0.7) since it understands semantic meaning.
    RULE_WEIGHT = 0.3
    LLM_WEIGHT = 0.7
    composite = {}
    dim_map = {
        "location_consistency": "rule_location",
        "time_appropriateness": "rule_time",
        "social_awareness": "rule_social",
        "memory_utilization": "rule_memory",
        "plan_adherence": "rule_plan",
    }

    for dim, rule_key in dim_map.items():
        llm_val = llm_scores.get(dim, 0)
        rule_val = rule_scores.get(rule_key, 1)
        composite[dim] = round(LLM_WEIGHT * llm_val + RULE_WEIGHT * rule_val, 4)

    composite["overall"] = sum(composite[d] for d in dim_map) / len(dim_map)
    composite["overall"] = round(composite["overall"], 4)
    composite["reasoning"] = llm_scores.get("reasoning", "")
    composite["rule_scores"] = rule_scores
    composite["llm_scores"] = {k: v for k, v in llm_scores.items() if k != "reasoning"}

    return composite


# =============================================================================
# RQ2: Factual Accuracy — judge chat response against ground truth
# =============================================================================

FACTUAL_SYSTEM = """You are a careful evaluator of factual accuracy in a simulated environment.

Given:
1. A user's question about a simulated environment
2. The GROUND TRUTH environment state (including agent states, locations, memories, events)
3. An AI's response

Classify each factual claim in the response into one of three categories:
- CORRECT: The claim is directly supported by the ground truth.
- CONTRADICTED: The claim directly contradicts the ground truth (e.g., wrong location, wrong person, wrong time).
- UNVERIFIABLE: The claim cannot be verified from the ground truth (e.g., subjective descriptions, spatial directions not in ground truth). These are NOT hallucinations.

SCORING RULES:
- precision = correct_claims / (correct_claims + contradicted_claims). If denominator is 0, precision = 0.5.
- Unverifiable claims are EXCLUDED from precision (they are neither hallucination nor verified).
- HOWEVER, also evaluate COMPLETENESS: did the response actually address the question with specific, useful information?
  - completeness = 1.0 if the response fully answers the question with specific details
  - completeness = 0.5 if the response partially answers or is vague
  - completeness = 0.0 if the response deflects, refuses, or says "I don't know" without attempting
- final accuracy = precision * 0.7 + completeness * 0.3 (weighted: accuracy matters more, but informativeness counts)

A good response is BOTH accurate AND informative. Being overly cautious ("I'm not sure") should NOT score high.
Being confidently wrong should score low.

Return ONLY a JSON object:
{"total_claims": N, "correct_claims": N, "contradicted_claims": N, "unverifiable_claims": N, "precision": 0.XX, "completeness": 0.XX, "accuracy": 0.XX, "hallucinations": ["list of CONTRADICTED claims only"], "reasoning": "..."}
"""


def judge_factual_accuracy(
    config: EvalConfig,
    query: str,
    ground_truth_state: Dict[str, Any],
    response: str,
    agent_name: str,
) -> Dict[str, Any]:
    """Judge factual accuracy of a chat response against ground truth."""
    client = _get_client(config)

    agent = next((a for a in ground_truth_state.get("agents", []) if a["name"] == agent_name), None)
    nearby = [a for a in ground_truth_state.get("agents", [])
              if a.get("location") == (agent or {}).get("location") and a["name"] != agent_name]

    # Include memories if available for richer ground truth
    agent_memories = (agent or {}).get('memories', [])
    memory_texts = []
    for m in agent_memories[:10]:
        if isinstance(m, dict):
            memory_texts.append(m.get('content', str(m)))
        else:
            memory_texts.append(str(m))

    user_prompt = f"""## User Query
"{query}" (asked by {agent_name})

## Ground Truth Environment State
- Time: {ground_truth_state.get('time', 'unknown')}
- {agent_name}'s Location: {(agent or {}).get('location', 'unknown')}
- {agent_name}'s Action: {(agent or {}).get('action', 'unknown')}
- Nearby Agents: {json.dumps([{'name': a['name'], 'action': a['action'], 'location': a['location']} for a in nearby])}
- All Agents: {json.dumps([{'name': a['name'], 'location': a['location'], 'action': a['action']} for a in ground_truth_state.get('agents', [])])}
- Recent Events: {json.dumps(ground_truth_state.get('events', [])[-10:])}
- {agent_name}'s Recent Memories: {json.dumps(memory_texts) if memory_texts else '[]'}
- {agent_name}'s Daily Plan: {json.dumps((agent or {}).get('daily_plan', (agent or {}).get('plan', [])))}

## AI Response
"{response}"

Classify each claim as CORRECT, CONTRADICTED, or UNVERIFIABLE based on the ground truth above."""

    raw = _call_judge(client, config, FACTUAL_SYSTEM, user_prompt)
    try:
        start = raw.index("{")
        end = raw.rindex("}") + 1
        return json.loads(raw[start:end])
    except (ValueError, json.JSONDecodeError):
        return {"error": "Failed to parse", "raw": raw}


# =============================================================================
# RQ3/RQ4: SOTOPIA 7-Dimension Social Evaluation
# =============================================================================

SOCIAL_SYSTEM = """You are an expert evaluator of social interactions in a multi-agent simulation.
Evaluate a conversation between two agents on 7 dimensions following the SOTOPIA framework.

For each dimension, provide a score within the given range and a brief justification.

Return ONLY a JSON object with this structure:
{
  "believability": {"score": X, "reason": "..."},
  "relationship": {"score": X, "reason": "..."},
  "knowledge": {"score": X, "reason": "..."},
  "secret": {"score": X, "reason": "..."},
  "social_rules": {"score": X, "reason": "..."},
  "financial": {"score": X, "reason": "..."},
  "goal": {"score": X, "reason": "..."},
  "overall_quality": X
}

Scoring ranges:
- believability: 0-10 (10 = perfectly natural and in-character)
- relationship: -5 to 5 (positive = relationship improved)
- knowledge: 0-10 (10 = excellent environmental awareness)
- secret: -10 to 0 (0 = no secrets leaked)
- social_rules: -10 to 0 (0 = perfect adherence to norms)
- financial: -5 to 5 (positive = material gains)
- goal: 0-10 (10 = fully achieved objective)
- overall_quality: 0-10
"""


def judge_social_interaction(
    config: EvalConfig,
    agent1_profile: Dict[str, Any],
    agent2_profile: Dict[str, Any],
    conversation: List[str],
    context: Dict[str, Any],
) -> Dict[str, Any]:
    """Judge a conversation using SOTOPIA 7-dimension framework."""
    client = _get_client(config)

    user_prompt = f"""## Agent 1 Profile
- Name: {agent1_profile.get('name')}
- Occupation: {agent1_profile.get('occupation')}
- Personality: {agent1_profile.get('personality')}
- Current Action: {agent1_profile.get('action')}

## Agent 2 Profile
- Name: {agent2_profile.get('name')}
- Occupation: {agent2_profile.get('occupation')}
- Personality: {agent2_profile.get('personality')}
- Current Action: {agent2_profile.get('action')}

## Context
- Location: {context.get('location', 'unknown')}
- Time: {context.get('time', 'unknown')}
- Relationship: {context.get('relationship', 'acquaintances')}

## Conversation
{chr(10).join(conversation)}

Evaluate this social interaction on all 7 SOTOPIA dimensions."""

    raw = _call_judge(client, config, SOCIAL_SYSTEM, user_prompt)
    try:
        start = raw.index("{")
        end = raw.rindex("}") + 1
        return json.loads(raw[start:end])
    except (ValueError, json.JSONDecodeError):
        return {"error": "Failed to parse", "raw": raw}


# =============================================================================
# RQ2 supplementary: Context Utilization scoring
# =============================================================================

CONTEXT_UTIL_SYSTEM = """You are evaluating how well an AI response utilized the environmental context that was provided.

Given:
1. The environmental context (location, nearby agents, memories, events)
2. The AI's response

Score from 0.0 to 1.0 how much of the provided context was meaningfully incorporated into the response.
0.0 = context completely ignored, 1.0 = all relevant context meaningfully used.

RULES:
- If the context is empty or minimal, utilization should be scored relative to what was available.
- Generic responses that could be written without the context score LOW.
- Responses that reference specific context elements (names, locations, times, events) score HIGH.

Return ONLY a JSON: {"utilization": 0.XX, "used_elements": ["list of context elements used"], "ignored_elements": ["list ignored"], "reasoning": "..."}
"""


def judge_context_utilization(
    config: EvalConfig,
    env_context: Dict[str, Any],
    response: str,
) -> Dict[str, Any]:
    """Judge how well the AI response utilized the provided env context."""
    client = _get_client(config)

    user_prompt = f"""## Environmental Context Provided
{json.dumps(env_context, indent=2, ensure_ascii=False)}

## AI Response
"{response}"

Evaluate context utilization."""

    raw = _call_judge(client, config, CONTEXT_UTIL_SYSTEM, user_prompt)
    try:
        start = raw.index("{")
        end = raw.rindex("}") + 1
        return json.loads(raw[start:end])
    except (ValueError, json.JSONDecodeError):
        return {"error": "Failed to parse", "raw": raw}
