# map_layer.py
# Map & Expedition layer for the Mars simulation game.
# stdlib only.

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, List, Optional, Tuple, Set


# ----------------------------- Map Model -------------------------------------

class TileType(str, Enum):
    PLAINS = "plains"
    ICE_FIELD = "ice_field"
    CANYON = "canyon"
    CRATER = "crater"
    METAL_DEPOSIT = "metal_deposit"


@dataclass
class Tile:
    x: int
    y: int
    t: TileType
    hazard: float                 # 0.0 .. 1.0
    discovered: bool = False
    visits: int = 0

    def char(self) -> str:
        # Single-char glyph for ASCII map
        return {
            TileType.PLAINS: ".",
            TileType.ICE_FIELD: "I",
            TileType.CANYON: "^",
            TileType.CRATER: "o",
            TileType.METAL_DEPOSIT: "M",
        }[self.t]


class WorldMap:
    """
    Seeded procedural grid centered at (0,0).
    Tiles are generated deterministically from (seed, x, y), so we only need
    to persist 'discovered' state and visit counters.
    """
    def __init__(self, w: int, h: int, seed: int = 0):
        if w % 2 == 0 or h % 2 == 0:
            raise ValueError("width and height must be odd so that (0,0) is the center.")
        self.w = w
        self.h = h
        self.seed = int(seed)
        # discovery is stored separately to avoid serializing full grid
        self._discovered: Set[Tuple[int, int]] = set()
        self._visits: Dict[Tuple[int, int], int] = {}

    # ---- Tile generation ----

    def _tile_rng(self, x: int, y: int) -> random.Random:
        # 2D hash → seed
        h = (x * 73856093) ^ (y * 19349663) ^ (self.seed * 83492791)
        r = random.Random()
        r.seed(h & 0xFFFFFFFF)
        return r

    def _gen_tile(self, x: int, y: int) -> Tile:
        r = self._tile_rng(x, y).random()
        # Bias: more plains; features more likely further from base
        dist = math.hypot(x, y)
        far = min(1.0, dist / max(self.w, self.h))
        # Base weights
        plains_w = 0.60 - 0.10 * far
        ice_w = 0.12 + 0.10 * far
        canyon_w = 0.10 + 0.05 * far
        crater_w = 0.10 + 0.05 * far
        metal_w = 0.08 + 0.05 * far
        total = plains_w + ice_w + canyon_w + crater_w + metal_w
        r_scaled = r * total
        if r_scaled < plains_w:
            t = TileType.PLAINS
            hz = 0.05 + 0.05 * far
        elif r_scaled < plains_w + ice_w:
            t = TileType.ICE_FIELD
            hz = 0.10 + 0.08 * far
        elif r_scaled < plains_w + ice_w + canyon_w:
            t = TileType.CANYON
            hz = 0.18 + 0.12 * far
        elif r_scaled < plains_w + ice_w + canyon_w + crater_w:
            t = TileType.CRATER
            hz = 0.14 + 0.08 * far
        else:
            t = TileType.METAL_DEPOSIT
            hz = 0.12 + 0.08 * far

        tile = Tile(x=x, y=y, t=t, hazard=min(0.9, hz))
        # restore discovery/visits if present
        if (x, y) in self._discovered:
            tile.discovered = True
            tile.visits = self._visits.get((x, y), 0)
        return tile

    def tile(self, x: int, y: int) -> Tile:
        cx = -(self.w // 2) <= x <= (self.w // 2)
        cy = -(self.h // 2) <= y <= (self.h // 2)
        if not (cx and cy):
            raise IndexError("coordinate outside world bounds")
        return self._gen_tile(x, y)

    def reveal(self, x: int, y: int, radius: int = 0):
        for yy in range(y - radius, y + radius + 1):
            for xx in range(x - radius, x + radius + 1):
                if not (-(self.w // 2) <= xx <= (self.w // 2)):
                    continue
                if not (-(self.h // 2) <= yy <= (self.h // 2)):
                    continue
                self._discovered.add((xx, yy))

    def visit(self, x: int, y: int):
        self._discovered.add((x, y))
        self._visits[(x, y)] = self._visits.get((x, y), 0) + 1

    def path_to(self, sx: int, sy: int, dx: int, dy: int) -> List[Tuple[int, int]]:
        """
        Simple Manhattan path; returns list including the destination but not the start.
        """
        path: List[Tuple[int, int]] = []
        x, y = sx, sy
        step_x = 1 if dx > x else -1
        step_y = 1 if dy > y else -1
        while x != dx:
            x += step_x
            path.append((x, y))
        while y != dy:
            y += step_y
            path.append((x, y))
        return path

    def ascii_map(self, center_x: int = 0, center_y: int = 0, view_radius: int = 6) -> str:
        """
        Render an ASCII minimap centered around (center_x, center_y).
        Unknown tiles are shown as '░'. Colony at (0,0) is 'C'.
        """
        lines: List[str] = []
        ymin = max(-(self.h // 2), center_y - view_radius)
        ymax = min((self.h // 2), center_y + view_radius)
        xmin = max(-(self.w // 2), center_x - view_radius)
        xmax = min((self.w // 2), center_x + view_radius)
        header = "   " + "".join(f"{x:>2}" for x in range(xmin, xmax + 1))
        lines.append(header)
        for y in range(ymin, ymax + 1):
            row = [f"{y:>2} "]
            for x in range(xmin, xmax + 1):
                if (x, y) == (0, 0):
                    row.append(" C")
                elif (x, y) in self._discovered:
                    row.append(" " + self.tile(x, y).char())
                else:
                    row.append(" ░")
            lines.append("".join(row))
        lines.append("\nLegend: C=Colony  .=Plains  I=Ice  ^=Canyon  o=Crater  M=Metal  ░=Unknown")
        return "\n".join(lines)

    # ---- Serialization ----

    def to_dict(self) -> dict:
        return {
            "w": self.w,
            "h": self.h,
            "seed": self.seed,
            "discovered": list([x, y] for (x, y) in sorted(self._discovered)),
            "visits": {f"{x},{y}": v for (x, y), v in self._visits.items()},
        }

    @classmethod
    def from_dict(cls, data: dict) -> "WorldMap":
        wm = cls(int(data["w"]), int(data["h"]), int(data.get("seed", 0)))
        wm._discovered = set((int(x), int(y)) for x, y in data.get("discovered", []))
        wm._visits = {}
        for k, v in data.get("visits", {}).items():
            x_s, y_s = k.split(",")
            wm._visits[(int(x_s), int(y_s))] = int(v)
        return wm


# --------------------------- Expedition System --------------------------------

class ExpeditionStatus(str, Enum):
    PREP = "prep"
    TRAVEL_OUT = "travel_out"
    GATHER = "gather"
    RETURNING = "returning"
    COMPLETE = "complete"
    ABORTED = "aborted"


@dataclass
class Expedition:
    world: WorldMap
    dest: Tuple[int, int]
    team_size: int = 2
    use_rover: bool = True
    seed: int = 0

    # runtime
    status: ExpeditionStatus = ExpeditionStatus.PREP
    path: List[Tuple[int, int]] = field(default_factory=list)
    step_index: int = 0
    cargo: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        if not self.path:
            self.path = self.world.path_to(0, 0, self.dest[0], self.dest[1])

    @property
    def active(self) -> bool:
        return self.status in {ExpeditionStatus.PREP, ExpeditionStatus.TRAVEL_OUT, ExpeditionStatus.GATHER,
                               ExpeditionStatus.RETURNING}

    def speed_steps_per_sol(self) -> float:
        # Rover halves the time (~2 steps per sol vs 1 on foot).
        return 2.0 if self.use_rover else 1.0

    def _rng(self) -> random.Random:
        r = random.Random()
        # Seed based on expedition seed + current phase + step
        phase_n = {
            ExpeditionStatus.PREP: 1,
            ExpeditionStatus.TRAVEL_OUT: 2,
            ExpeditionStatus.GATHER: 3,
            ExpeditionStatus.RETURNING: 4,
            ExpeditionStatus.COMPLETE: 5,
            ExpeditionStatus.ABORTED: 6
        }[self.status]
        r.seed((self.seed * 10007) ^ (phase_n * 911) ^ (self.step_index * 37))
        return r

    def _hazard_event(self, tile: Tile) -> Optional[str]:
        """
        Returns a log string if something bad happens; otherwise None.
        """
        # Base daily hazard chance scaled by tile hazard and team size (bigger team -> safer).
        base = 0.05 + tile.hazard * 0.25
        mitigation = 0.02 * max(0, self.team_size - 2)
        p = max(0.0, min(0.6, base - mitigation))
        if self._rng().random() < p:
            # choose an event flavor
            events = [
                "minor suit tear patched on the fly (-small morale).",
                "dust devil knocked gear around (-small condition).",
                "navigation detour added time.",
                "crew fatigue set in (-small health).",
            ]
            return events[int(self._rng().random() * len(events)) % len(events)]
        return None

    def _gather_from_tile(self, tile: Tile) -> Dict[str, float]:
        r = self._rng()
        k = self.team_size
        loot: Dict[str, float] = {}
        if tile.t == TileType.ICE_FIELD:
            loot["water"] = r.randint(6, 12) * k
        elif tile.t == TileType.METAL_DEPOSIT:
            loot["metals"] = r.randint(5, 10) * k
        elif tile.t == TileType.CANYON:
            # odds of polymers (salvaged volatiles/plastics), sometimes nothing
            if r.random() < 0.6:
                loot["polymers"] = r.randint(1, 3) * k
        elif tile.t == TileType.CRATER:
            # small chance for fuel canister (met impacts)
            if r.random() < 0.20:
                loot["fuel"] = r.randint(1, 2)
            else:
                loot["metals"] = r.randint(2, 5) * k
        else:  # plains
            if r.random() < 0.15:
                loot["food"] = r.randint(1, 3) * k  # snacks from field kits
        return loot

    def tick(self, colony, env) -> Tuple[List[str], Dict[str, float], bool]:
        """
        Advance the expedition by one sol.
        Returns (log_lines, delivered_cargo, finished_bool).
        """
        logs: List[str] = []
        delivered: Dict[str, float] = {}

        # PREP → TRAVEL_OUT
        if self.status == ExpeditionStatus.PREP:
            self.status = ExpeditionStatus.TRAVEL_OUT
            self.step_index = 0
            self.world.reveal(self.dest[0], self.dest[1], radius=1)
            logs.append(f"🧭 Expedition departed toward {self.dest}.")
            return logs, delivered, False

        # Move along path
        if self.status == ExpeditionStatus.TRAVEL_OUT:
            steps = self.speed_steps_per_sol()
            for _ in range(math.ceil(steps)):
                if self.step_index >= len(self.path):
                    break
                cx, cy = self.path[self.step_index]
                tile = self.world.tile(cx, cy)
                self.world.reveal(cx, cy, radius=0)
                self.world.visit(cx, cy)
                ev = self._hazard_event(tile)
                if ev:
                    logs.append(f"⚠️ Travel event at {cx},{cy} ({tile.t.value}): {ev}")
                self.step_index += 1
            if self.step_index >= len(self.path):
                self.status = ExpeditionStatus.GATHER
                logs.append(f"⛏️ Reached destination {self.dest}. Gathering...")
                tile = self.world.tile(self.dest[0], self.dest[1])
                gain = self._gather_from_tile(tile)
                for k, v in gain.items():
                    self.cargo[k] = self.cargo.get(k, 0.0) + float(v)
                if gain:
                    loot_s = ", ".join(f"{k}+{int(v)}" for k, v in gain.items())
                    logs.append(f"📦 Findings: {loot_s}")
                else:
                    logs.append("📦 Findings: none this time.")
                self.status = ExpeditionStatus.RETURNING
                # flip path to go home, reuse step_index as 'step on return'
                self.path = list(reversed(self.path))
                self.step_index = 0
            return logs, delivered, False

        if self.status == ExpeditionStatus.RETURNING:
            steps = self.speed_steps_per_sol()
            for _ in range(math.ceil(steps)):
                if self.step_index >= len(self.path):
                    break
                cx, cy = self.path[self.step_index]
                tile = self.world.tile(cx, cy)
                ev = self._hazard_event(tile)
                if ev:
                    logs.append(f"⚠️ Return-leg event at {cx},{cy} ({tile.t.value}): {ev}")
                self.step_index += 1
            if self.step_index >= len(self.path):
                self.status = ExpeditionStatus.COMPLETE
                delivered = dict(self.cargo)
                logs.append("🏠 Expedition returned to base.")
                return logs, delivered, True
            return logs, delivered, False

        # COMPLETE / ABORTED: nothing to do
        return logs, delivered, self.status in (ExpeditionStatus.COMPLETE, ExpeditionStatus.ABORTED)

    # ---- Serialization ----

    def to_dict(self) -> dict:
        return {
            "dest": list(self.dest),
            "team_size": self.team_size,
            "use_rover": self.use_rover,
            "seed": self.seed,
            "status": self.status.value,
            "path": [list(p) for p in self.path],
            "step_index": self.step_index,
            "cargo": dict(self.cargo),
        }

    @classmethod
    def from_dict(cls, world: WorldMap, data: dict) -> "Expedition":
        ex = cls(
            world=world,
            dest=(int(data["dest"][0]), int(data["dest"][1])),
            team_size=int(data.get("team_size", 2)),
            use_rover=bool(data.get("use_rover", True)),
            seed=int(data.get("seed", 0)),
        )
        ex.status = ExpeditionStatus(data.get("status", "prep"))
        ex.path = [tuple(map(int, p)) for p in data.get("path", [])]
        ex.step_index = int(data.get("step_index", 0))
        ex.cargo = {str(k): float(v) for k, v in data.get("cargo", {}).items()}
        return ex
