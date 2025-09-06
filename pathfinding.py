# pathfinding.py
from __future__ import annotations
from dataclasses import dataclass
from heapq import heappush, heappop
from typing import Callable, Dict, Iterable, List, Optional, Tuple
import math

Pos = Tuple[int, int]

@dataclass(frozen=True)
class PathResult:
    path: List[Pos]          # [start, ..., goal]
    cost: float              # sum of tile movement costs (incl. diagonals)
    expanded: int            # nodes expanded (for debugging/profile)

def _octile(a: Pos, b: Pos) -> float:
    # Heuristic for 8-directional grids (good balance of speed/quality)
    dx, dy = abs(a[0] - b[0]), abs(a[1] - b[1])
    # cost(orth)=1, cost(diag)=sqrt(2). Scale if your base cost != 1
    return (max(dx, dy) - min(dx, dy)) + math.sqrt(2) * min(dx, dy)

def astar(
    start: Pos,
    goal: Pos,
    passable: Callable[[Pos], bool],
    move_cost: Callable[[Pos, Pos], float],
    allow_diagonals: bool = True,
) -> Optional[PathResult]:
    """A* over a grid. passable() checks destination cells; move_cost() returns
    the incremental cost from 'a' -> 'b'. Returns None if no path exists.
    """

    if start == goal:
        return PathResult([start], cost=0.0, expanded=0)

    # 8-neighbor or 4-neighbor moves
    steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if allow_diagonals:
        steps += [(-1, -1), (-1, 1), (1, -1), (1, 1)]

    openq: List[Tuple[float, float, Pos]] = []
    heappush(openq, (0.0, 0.0, start))
    g: Dict[Pos, float] = {start: 0.0}
    parent: Dict[Pos, Pos] = {}
    expanded = 0

    while openq:
        _, _, current = heappop(openq)
        expanded += 1
        if current == goal:
            # Reconstruct
            path: List[Pos] = [current]
            while current in parent:
                current = parent[current]
                path.append(current)
            path.reverse()
            return PathResult(path=path, cost=g[path[-1]], expanded=expanded)

        cx, cy = current
        for dx, dy in steps:
            nx, ny = cx + dx, cy + dy
            nxt = (nx, ny)
            if not passable(nxt):
                continue

            step = move_cost((cx, cy), nxt)
            if math.isinf(step) or step <= 0:
                continue

            new_g = g[current] + step
            if new_g < g.get(nxt, float("inf")):
                g[nxt] = new_g
                parent[nxt] = current
                h = _octile(nxt, goal) if allow_diagonals else (abs(nx - goal[0]) + abs(ny - goal[1]))
                f = new_g + h
                heappush(openq, (f, new_g, nxt))

    return None
