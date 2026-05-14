"""
Social network analysis and emergent behavior detection for AURA Town.

Provides:
1. SocialNetworkAnalyzer - builds interaction graphs, computes network metrics
2. BehaviorPatternDetector - detects emergent social patterns
"""

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple


@dataclass
class SocialEdge:
    """A directed interaction between two agents."""
    agent_a: str
    agent_b: str
    location: str
    time: str
    interaction_type: str  # conversation, co-presence, cooperation


@dataclass
class NetworkMetrics:
    """Summary metrics for the social network."""
    num_agents: int = 0
    num_edges: int = 0
    density: float = 0.0
    avg_degree: float = 0.0
    per_agent_degree: Dict[str, int] = field(default_factory=dict)
    conversation_frequency: Dict[str, int] = field(default_factory=dict)
    location_distribution: Dict[str, int] = field(default_factory=dict)
    interaction_timeline: List[Dict[str, Any]] = field(default_factory=list)


class SocialNetworkAnalyzer:
    """Build and analyze the agent social interaction graph."""

    def __init__(self) -> None:
        self.edges: List[SocialEdge] = []
        self.agents: Set[str] = set()

    def add_event(self, event: Dict[str, Any]) -> None:
        """Process a simulation event and extract social interactions."""
        etype = event.get("type", "")
        agent = event.get("agent", "")
        details = event.get("details", {})

        if etype == "conversation":
            dialogue = details.get("dialogue", [])
            speakers = set()
            for line in dialogue:
                if ":" in line:
                    speakers.add(line.split(":")[0].strip())
            speaker_list = sorted(speakers)
            if len(speaker_list) >= 2:
                self.agents.update(speaker_list)
                self.edges.append(SocialEdge(
                    agent_a=speaker_list[0],
                    agent_b=speaker_list[1],
                    location=details.get("location", "unknown"),
                    time=event.get("time", ""),
                    interaction_type="conversation",
                ))
        elif etype == "action" and agent:
            self.agents.add(agent)

    def add_co_presence(self, agents_at_location: Dict[str, List[str]]) -> None:
        """Record co-presence from a state snapshot (agents at the same location)."""
        for location, names in agents_at_location.items():
            for i, a in enumerate(names):
                for b in names[i + 1:]:
                    self.agents.update([a, b])
                    self.edges.append(SocialEdge(
                        agent_a=min(a, b),
                        agent_b=max(a, b),
                        location=location,
                        time="",
                        interaction_type="co-presence",
                    ))

    def compute_metrics(self) -> NetworkMetrics:
        """Compute social network metrics."""
        n = len(self.agents)
        conversations = [e for e in self.edges if e.interaction_type == "conversation"]

        # Degree: number of unique conversation partners per agent
        adj: Dict[str, Set[str]] = defaultdict(set)
        for e in conversations:
            adj[e.agent_a].add(e.agent_b)
            adj[e.agent_b].add(e.agent_a)

        per_agent_degree = {a: len(adj.get(a, set())) for a in self.agents}

        # Conversation frequency per pair
        pair_count: Counter = Counter()
        for e in conversations:
            pair = tuple(sorted([e.agent_a, e.agent_b]))
            pair_count[pair] += 1

        # Location distribution of conversations
        loc_dist: Counter = Counter()
        for e in conversations:
            loc_dist[e.location] += 1

        max_edges = n * (n - 1) / 2 if n > 1 else 1
        density = len(set(
            tuple(sorted([e.agent_a, e.agent_b])) for e in conversations
        )) / max_edges if max_edges > 0 else 0

        return NetworkMetrics(
            num_agents=n,
            num_edges=len(conversations),
            density=round(density, 4),
            avg_degree=round(sum(per_agent_degree.values()) / max(n, 1), 2),
            per_agent_degree=per_agent_degree,
            conversation_frequency={f"{a}-{b}": c for (a, b), c in pair_count.items()},
            location_distribution=dict(loc_dist),
        )

    def to_dict(self) -> Dict[str, Any]:
        metrics = self.compute_metrics()
        return {
            "num_agents": metrics.num_agents,
            "num_edges": metrics.num_edges,
            "density": metrics.density,
            "avg_degree": metrics.avg_degree,
            "per_agent_degree": metrics.per_agent_degree,
            "conversation_frequency": metrics.conversation_frequency,
            "location_distribution": metrics.location_distribution,
        }


@dataclass
class EmergentBehavior:
    """A detected emergent social behavior."""
    behavior_type: str  # routine_adaptation, information_propagation, group_formation, collaboration, conflict_resolution
    description: str
    agents_involved: List[str]
    evidence: List[str]
    confidence: float  # 0-1


class BehaviorPatternDetector:
    """Detect emergent social behavior patterns from event sequences."""

    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []
        self.agent_actions: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.conversations: List[Dict[str, Any]] = []

    def add_event(self, event: Dict[str, Any]) -> None:
        self.events.append(event)
        agent = event.get("agent", "")
        etype = event.get("type", "")
        if agent:
            self.agent_actions[agent].append(event)
        if etype == "conversation":
            self.conversations.append(event)

    def detect_routine_adaptation(self) -> List[EmergentBehavior]:
        """Detect agents adapting their routines based on environment changes."""
        behaviors = []
        for agent, actions in self.agent_actions.items():
            if len(actions) < 5:
                continue
            # Check for location pattern changes (e.g., agent starts visiting new places)
            locations = [a.get("details", {}).get("location", a.get("description", "")) for a in actions]
            first_half = set(locations[:len(locations) // 2])
            second_half = set(locations[len(locations) // 2:])
            new_places = second_half - first_half
            if new_places:
                behaviors.append(EmergentBehavior(
                    behavior_type="routine_adaptation",
                    description=f"{agent} adapted routine to include new locations: {', '.join(new_places)}",
                    agents_involved=[agent],
                    evidence=[f"First half locations: {first_half}", f"Second half locations: {second_half}"],
                    confidence=min(len(new_places) / max(len(second_half), 1), 1.0),
                ))
        return behaviors

    def detect_information_propagation(self) -> List[EmergentBehavior]:
        """Detect information spreading through conversations."""
        behaviors = []
        # Track topics mentioned in conversations over time
        topic_agents: Dict[str, List[Tuple[str, str]]] = defaultdict(list)

        for conv in self.conversations:
            details = conv.get("details", {})
            dialogue = details.get("dialogue", [])
            speakers = set()
            for line in dialogue:
                if ":" in line:
                    speakers.add(line.split(":")[0].strip())

            # Simple topic extraction: look for common nouns/phrases
            text = " ".join(dialogue).lower()
            for keyword in ["event", "festival", "meeting", "news", "plan", "weather", "new"]:
                if keyword in text:
                    for speaker in speakers:
                        topic_agents[keyword].append((speaker, conv.get("time", "")))

        for topic, mentions in topic_agents.items():
            unique_agents = set(a for a, _ in mentions)
            if len(unique_agents) >= 3:
                behaviors.append(EmergentBehavior(
                    behavior_type="information_propagation",
                    description=f"Topic '{topic}' spread across {len(unique_agents)} agents",
                    agents_involved=sorted(unique_agents),
                    evidence=[f"{a} mentioned at {t}" for a, t in mentions[:5]],
                    confidence=min(len(unique_agents) / 5, 1.0),
                ))
        return behaviors

    def detect_group_formation(self) -> List[EmergentBehavior]:
        """Detect agents forming groups (frequent co-location or conversation clusters)."""
        behaviors = []
        # Pair frequency
        pair_count: Counter = Counter()
        for conv in self.conversations:
            details = conv.get("details", {})
            dialogue = details.get("dialogue", [])
            speakers = set()
            for line in dialogue:
                if ":" in line:
                    speakers.add(line.split(":")[0].strip())
            speaker_list = sorted(speakers)
            if len(speaker_list) >= 2:
                pair_count[tuple(speaker_list[:2])] += 1

        # Pairs with >= 3 conversations form a group
        frequent_pairs = [(pair, count) for pair, count in pair_count.items() if count >= 3]
        if frequent_pairs:
            # Merge overlapping pairs into groups
            groups: List[Set[str]] = []
            for (a, b), _ in frequent_pairs:
                merged = False
                for group in groups:
                    if a in group or b in group:
                        group.update([a, b])
                        merged = True
                        break
                if not merged:
                    groups.append({a, b})

            for group in groups:
                if len(group) >= 2:
                    behaviors.append(EmergentBehavior(
                        behavior_type="group_formation",
                        description=f"Social group formed: {', '.join(sorted(group))}",
                        agents_involved=sorted(group),
                        evidence=[f"{a}-{b}: {c} conversations" for (a, b), c in frequent_pairs
                                  if a in group or b in group],
                        confidence=min(sum(c for (a, b), c in frequent_pairs
                                           if a in group or b in group) / 10, 1.0),
                    ))
        return behaviors

    def detect_collaboration(self) -> List[EmergentBehavior]:
        """Detect collaborative behavior (agents working toward shared goals)."""
        behaviors = []
        # Look for conversations mentioning cooperation keywords
        for conv in self.conversations:
            details = conv.get("details", {})
            dialogue = details.get("dialogue", [])
            text = " ".join(dialogue).lower()

            coop_keywords = ["together", "help", "collaborate", "plan", "organize", "team", "share"]
            if any(kw in text for kw in coop_keywords):
                speakers = set()
                for line in dialogue:
                    if ":" in line:
                        speakers.add(line.split(":")[0].strip())
                if len(speakers) >= 2:
                    behaviors.append(EmergentBehavior(
                        behavior_type="collaboration",
                        description=f"Collaborative interaction between {', '.join(sorted(speakers))}",
                        agents_involved=sorted(speakers),
                        evidence=[line for line in dialogue if any(kw in line.lower() for kw in coop_keywords)][:3],
                        confidence=0.7,
                    ))
        return behaviors

    def detect_conflict_resolution(self) -> List[EmergentBehavior]:
        """Detect conflict and resolution patterns in conversations."""
        behaviors = []
        conflict_words = ["disagree", "sorry", "misunderstand", "problem", "issue", "concern", "argue"]
        resolution_words = ["agree", "understand", "sorry", "resolve", "compromise", "okay", "fine"]

        for conv in self.conversations:
            details = conv.get("details", {})
            dialogue = details.get("dialogue", [])
            text = " ".join(dialogue).lower()

            has_conflict = any(w in text for w in conflict_words)
            has_resolution = any(w in text for w in resolution_words)

            if has_conflict and has_resolution:
                speakers = set()
                for line in dialogue:
                    if ":" in line:
                        speakers.add(line.split(":")[0].strip())
                if len(speakers) >= 2:
                    behaviors.append(EmergentBehavior(
                        behavior_type="conflict_resolution",
                        description=f"Conflict resolved between {', '.join(sorted(speakers))}",
                        agents_involved=sorted(speakers),
                        evidence=[line for line in dialogue
                                  if any(w in line.lower() for w in conflict_words + resolution_words)][:3],
                        confidence=0.6,
                    ))
        return behaviors

    def detect_all(self) -> List[EmergentBehavior]:
        """Run all detectors and return combined results."""
        all_behaviors = []
        all_behaviors.extend(self.detect_routine_adaptation())
        all_behaviors.extend(self.detect_information_propagation())
        all_behaviors.extend(self.detect_group_formation())
        all_behaviors.extend(self.detect_collaboration())
        all_behaviors.extend(self.detect_conflict_resolution())
        return all_behaviors

    def to_summary(self) -> Dict[str, Any]:
        """Return a JSON-serialisable summary of all detected behaviors."""
        behaviors = self.detect_all()
        by_type: Dict[str, List[Dict]] = defaultdict(list)
        for b in behaviors:
            by_type[b.behavior_type].append({
                "description": b.description,
                "agents": b.agents_involved,
                "evidence": b.evidence,
                "confidence": b.confidence,
            })
        return {
            "total_behaviors": len(behaviors),
            "by_type": {k: {"count": len(v), "instances": v} for k, v in by_type.items()},
            "behavior_type_counts": {k: len(v) for k, v in by_type.items()},
        }
