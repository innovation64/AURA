"""Frontier-based spatial exploration with A* pathfinding for AURA Town agents."""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from .chunks import CHUNK_SIZE


@dataclass
class ExplorationGoal:
    """Represents an agent's current exploration objective."""

    goal_type: str  # "frontier", "resource", "curiosity", "task"
    target_chunk: Tuple[int, int]
    path: List[Tuple[int, int]] = field(default_factory=list)
    priority: float = 1.0
    reason: str = ""
    target_world: Optional[Tuple[int, int]] = None  # world-coord target

    def __post_init__(self) -> None:
        if self.target_world is None:
            cx, cy = self.target_chunk
            self.target_world = (cx * CHUNK_SIZE + CHUNK_SIZE // 2,
                                 cy * CHUNK_SIZE + CHUNK_SIZE // 2)


class FrontierDetector:
    """Detects unexplored chunks adjacent to explored territory."""

    @staticmethod
    def get_frontiers(explored_chunks: Set[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Return chunk coordinates that are adjacent to explored territory but not yet explored.

        Results are sorted by distance to centroid of explored area for a natural
        expanding-outward exploration pattern.
        """
        if not explored_chunks:
            return [(0, 0)]

        frontiers: Set[Tuple[int, int]] = set()
        for cx, cy in explored_chunks:
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                neighbor = (cx + dx, cy + dy)
                if neighbor not in explored_chunks:
                    frontiers.add(neighbor)

        if not frontiers:
            return []

        # Sort by distance to centroid for natural expansion
        avg_x = sum(c[0] for c in explored_chunks) / len(explored_chunks)
        avg_y = sum(c[1] for c in explored_chunks) / len(explored_chunks)
        return sorted(frontiers, key=lambda f: (f[0] - avg_x) ** 2 + (f[1] - avg_y) ** 2)

    @staticmethod
    def get_frontiers_near(
        explored_chunks: Set[Tuple[int, int]],
        agent_x: int,
        agent_y: int,
        max_results: int = 5,
    ) -> List[Tuple[int, int]]:
        """Get frontier chunks sorted by distance to the agent's position."""
        frontiers = FrontierDetector.get_frontiers(explored_chunks)
        agent_cx = agent_x // CHUNK_SIZE
        agent_cy = agent_y // CHUNK_SIZE
        frontiers.sort(key=lambda f: (f[0] - agent_cx) ** 2 + (f[1] - agent_cy) ** 2)
        return frontiers[:max_results]


class AStarPathfinder:
    """A* grid pathfinding that avoids building interiors."""

    @staticmethod
    def find_path(
        start: Tuple[int, int],
        goal: Tuple[int, int],
        world: Any,
        max_steps: int = 200,
    ) -> List[Tuple[int, int]]:
        """Find a path from start to goal on the world grid, avoiding buildings.

        Returns a list of (x, y) world coordinates from start (exclusive) to goal (inclusive).
        Returns empty list if no path found within max_steps.
        """
        sx, sy = start
        gx, gy = goal

        # Quick check: if start == goal, no path needed
        if sx == gx and sy == gy:
            return []

        # Build obstacle set from world locations (building interiors)
        obstacles: Set[Tuple[int, int]] = set()
        for loc in world.locations:
            # Don't block the entrance (center-bottom cell)
            entrance = loc.center
            for dy in range(loc.height):
                for dx in range(loc.width):
                    cell = (loc.x + dx, loc.y + dy)
                    if cell != entrance:
                        obstacles.add(cell)

        # A* with Manhattan distance heuristic
        def heuristic(x: int, y: int) -> int:
            return abs(x - gx) + abs(y - gy)

        open_set: list = []  # (f_score, counter, x, y)
        counter = 0
        heapq.heappush(open_set, (heuristic(sx, sy), counter, sx, sy))
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        g_score: Dict[Tuple[int, int], int] = {(sx, sy): 0}
        closed: Set[Tuple[int, int]] = set()

        while open_set and len(closed) < max_steps:
            _, _, cx, cy = heapq.heappop(open_set)

            if (cx, cy) in closed:
                continue
            closed.add((cx, cy))

            if cx == gx and cy == gy:
                # Reconstruct path
                path = []
                current = (gx, gy)
                while current != (sx, sy):
                    path.append(current)
                    current = came_from[current]
                path.reverse()
                return path

            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = cx + dx, cy + dy
                if (nx, ny) in closed:
                    continue
                if (nx, ny) in obstacles:
                    continue

                new_g = g_score[(cx, cy)] + 1
                if new_g < g_score.get((nx, ny), float("inf")):
                    g_score[(nx, ny)] = new_g
                    f = new_g + heuristic(nx, ny)
                    came_from[(nx, ny)] = (cx, cy)
                    counter += 1
                    heapq.heappush(open_set, (f, counter, nx, ny))

        # No path found within budget
        return []


class SpatialExplorer:
    """Manages exploration goals and path execution for agents."""

    def __init__(self, world: Any) -> None:
        self._world = world

    def select_exploration_target(
        self,
        agent_x: int,
        agent_y: int,
        explored_chunks: Set[Tuple[int, int]],
        goal_hint: Optional[str] = None,
        preferred_biomes: Optional[List[str]] = None,
    ) -> Optional[ExplorationGoal]:
        """Select the best exploration target for an agent.

        Args:
            agent_x, agent_y: Agent's current world position
            explored_chunks: Set of already-explored chunk coordinates
            goal_hint: Optional text hint (e.g. "find a river", "go north")
            preferred_biomes: Optional list of preferred biome types
        """
        frontiers = FrontierDetector.get_frontiers_near(
            explored_chunks, agent_x, agent_y, max_results=10
        )
        if not frontiers:
            return None

        # Score frontiers
        best_frontier = None
        best_score = -1.0

        for fx, fy in frontiers:
            score = 1.0
            dist = abs(fx - agent_x // CHUNK_SIZE) + abs(fy - agent_y // CHUNK_SIZE)
            # Prefer closer frontiers (inverse distance)
            score += max(0, 5 - dist) * 0.3

            # Biome preference bonus
            if preferred_biomes:
                from .chunks import assign_biome
                biome = assign_biome(fx, fy)
                if biome in preferred_biomes:
                    score += 2.0

            # Direction hint matching
            if goal_hint:
                hint_lower = goal_hint.lower()
                agent_cx = agent_x // CHUNK_SIZE
                agent_cy = agent_y // CHUNK_SIZE
                if "north" in hint_lower and fy < agent_cy:
                    score += 1.5
                elif "south" in hint_lower and fy > agent_cy:
                    score += 1.5
                elif "east" in hint_lower and fx > agent_cx:
                    score += 1.5
                elif "west" in hint_lower and fx < agent_cx:
                    score += 1.5
                if "river" in hint_lower or "water" in hint_lower:
                    from .chunks import assign_biome
                    if assign_biome(fx, fy) == "riverside":
                        score += 3.0
                if "forest" in hint_lower or "tree" in hint_lower:
                    from .chunks import assign_biome
                    if assign_biome(fx, fy) == "forest":
                        score += 3.0
                if "mountain" in hint_lower or "peak" in hint_lower:
                    from .chunks import assign_biome
                    if assign_biome(fx, fy) == "mountain":
                        score += 3.0

            if score > best_score:
                best_score = score
                best_frontier = (fx, fy)

        if best_frontier is None:
            return None

        # Compute target world coordinate (center of target chunk)
        target_wx = best_frontier[0] * CHUNK_SIZE + CHUNK_SIZE // 2
        target_wy = best_frontier[1] * CHUNK_SIZE + CHUNK_SIZE // 2

        # Find A* path
        path = AStarPathfinder.find_path(
            (agent_x, agent_y),
            (target_wx, target_wy),
            self._world,
            max_steps=300,
        )

        goal_type = "frontier"
        reason = f"Exploring frontier chunk ({best_frontier[0]}, {best_frontier[1]})"
        if goal_hint:
            goal_type = "task"
            reason = f"Exploring toward: {goal_hint}"

        return ExplorationGoal(
            goal_type=goal_type,
            target_chunk=best_frontier,
            path=path,
            priority=best_score,
            reason=reason,
            target_world=(target_wx, target_wy),
        )

    @staticmethod
    def step_toward_goal(
        agent_x: int,
        agent_y: int,
        goal: ExplorationGoal,
        speed: int = 3,
    ) -> Tuple[int, int]:
        """Move the agent one step along the exploration path.

        Returns the new (x, y) position.
        """
        if goal.path:
            # Follow the pre-computed A* path
            steps_taken = 0
            new_x, new_y = agent_x, agent_y
            while goal.path and steps_taken < speed:
                next_pos = goal.path[0]
                dx = next_pos[0] - new_x
                dy = next_pos[1] - new_y
                step_cost = abs(dx) + abs(dy)
                if step_cost <= speed - steps_taken:
                    new_x, new_y = next_pos
                    goal.path.pop(0)
                    steps_taken += step_cost
                else:
                    # Partial move toward next waypoint
                    if abs(dx) > 0:
                        move = min(abs(dx), speed - steps_taken)
                        new_x += move if dx > 0 else -move
                        steps_taken += move
                    if abs(dy) > 0 and steps_taken < speed:
                        move = min(abs(dy), speed - steps_taken)
                        new_y += move if dy > 0 else -move
                        steps_taken += move
                    break
            return (new_x, new_y)

        # No path: move directly toward target
        if goal.target_world:
            tx, ty = goal.target_world
        else:
            tx = goal.target_chunk[0] * CHUNK_SIZE + CHUNK_SIZE // 2
            ty = goal.target_chunk[1] * CHUNK_SIZE + CHUNK_SIZE // 2

        dx = tx - agent_x
        dy = ty - agent_y
        total = abs(dx) + abs(dy)
        if total == 0:
            return (agent_x, agent_y)

        new_x, new_y = agent_x, agent_y
        remaining = speed
        if dx != 0:
            step = min(abs(dx), remaining) * (1 if dx > 0 else -1)
            new_x += step
            remaining -= abs(step)
        if dy != 0 and remaining > 0:
            step = min(abs(dy), remaining) * (1 if dy > 0 else -1)
            new_y += step

        return (new_x, new_y)

    @staticmethod
    def is_goal_reached(
        agent_x: int,
        agent_y: int,
        goal: ExplorationGoal,
        threshold: int = 3,
    ) -> bool:
        """Check if the agent has reached close enough to the exploration goal."""
        if goal.target_world:
            tx, ty = goal.target_world
        else:
            tx = goal.target_chunk[0] * CHUNK_SIZE + CHUNK_SIZE // 2
            ty = goal.target_chunk[1] * CHUNK_SIZE + CHUNK_SIZE // 2
        return abs(agent_x - tx) + abs(agent_y - ty) <= threshold
