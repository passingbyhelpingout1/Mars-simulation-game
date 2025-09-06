# mars_colony.py
# A compact, single-file starter for a Mars Colony simulation game.
# Python 3.8+ / stdlib only. (Optional: pygame for a tile-based map viewer.)

from __future__ import annotations

import json
import math
import random
import sys
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, List, Optional, Tuple

# --------------------------- Optional GFX (pygame) ----------------------------
# The game runs fully in the terminal. If pygame is installed, you can view the
# world map with a simple tile renderer (pan with arrows, zoom with +/-).
try:
    import pygame  # type: ignore
    _HAS_PYGAME = True
except Exception:  # pragma: no cover
    pygame = None  # type: ignore
    _HAS_PYGAME = False


def _color_for_char(ch: str) -> Tuple[int, int, int]:
    """
    Heuristic palette for common map ASCII glyphs. Unknown chars get a neutral tint.
    """
    base = {
        " ": (12, 12, 14),   # void / fog
        ".": (110, 78, 55),  # regolith
        ",": (120, 86, 60),
        "#": (120, 120, 130),  # rock / ridge
        "H": (70, 110, 210),   # habitat
        "S": (70, 170, 180),   # solar
        "R": (240, 130, 40),   # rover
        "W": (50, 120, 220),   # water / ice
        "F": (70, 160, 70),    # food-ish
        "M": (150, 150, 170),  # metals
        "?": (40, 40, 55),     # unknown / unrevealed
        "X": (160, 60, 160),   # anomaly
        "+": (60, 60, 70),
        "-": (60, 60, 70),
        "|": (60, 60, 70),
        "*": (180, 170, 80),
        "~": (80, 90, 140),
    }
    if ch in base:
        return base[ch]
    if ch.isalpha():
        return (120, 120, 180)
    if ch.isdigit():
        return (85, 85, 95)
    return (100, 105, 115)


if _HAS_PYGAME:

    class PygameMapViewer:
        """
        Minimal tile renderer that draws a 2D character grid as colored tiles.
        - Pan with arrow keys
        - Zoom with +/- (or = on US keyboards)
        - Close with Esc or Enter
        """
        def __init__(self, tile_px: int = 24, window_size: Tuple[int, int] = (1024, 768)) -> None:
            pygame.init()
            pygame.display.set_caption("Mars Map")
            self.screen = pygame.display.set_mode(window_size)
            self.clock = pygame.time.Clock()
            try:
                pygame.font.init()
                self.font = pygame.font.SysFont(None, 16)
            except Exception:
                self.font = None
            self.tile_px = tile_px
            self.zoom = 1.0
            self.cam_x = 0
            self.cam_y = 0

        def _handle_input(self) -> bool:
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    return False
                if e.type == pygame.KEYDOWN and e.key in (pygame.K_ESCAPE, pygame.K_RETURN):
                    return False
            keys = pygame.key.get_pressed()
            # pan speed in pixels per second
            px = int(420 * max(1, self.clock.get_time()) / 1000)
            if keys[pygame.K_LEFT]:
                self.cam_x -= px
            if keys[pygame.K_RIGHT]:
                self.cam_x += px
            if keys[pygame.K_UP]:
                self.cam_y -= px
            if keys[pygame.K_DOWN]:
                self.cam_y += px
            if keys[pygame.K_MINUS] or keys[pygame.K_KP_MINUS]:
                self.zoom = max(0.5, self.zoom - 0.05)
            if keys[pygame.K_EQUALS] or keys[pygame.K_KP_PLUS]:
                self.zoom = min(3.0, self.zoom + 0.05)
            return True

        def _draw(self, grid: List[List[str]], hud_text: str = "") -> None:
            ts = max(4, int(self.tile_px * self.zoom))
            self.screen.fill((8, 8, 10))
            h = len(grid)
            w = len(grid[0]) if h else 0

            # visible rect
            start_x = max(0, self.cam_x // ts)
            start_y = max(0, self.cam_y // ts)
            tiles_x = self.screen.get_width() // ts + 2
            tiles_y = self.screen.get_height() // ts + 2

            ypix = - (self.cam_y % ts)
            for gy in range(start_y, min(start_y + tiles_y, h)):
                xpix = - (self.cam_x % ts)
                row = grid[gy]
                for gx in range(start_x, min(start_x + tiles_x, w)):
                    ch = row[gx]
                    pygame.draw.rect(self.screen, _color_for_char(ch), (xpix, ypix, ts, ts))
                    # Optional letter overlay for clarity at larger sizes
                    if self.font and ts >= 18 and ch.strip():
                        surf = self.font.render(ch, True, (230, 230, 230))
                        self.screen.blit(surf, (xpix + 3, ypix + 2))
                    xpix += ts
                ypix += ts

            if self.font:
                bar = self.font.render("←↑↓→ pan    +/- zoom    Esc/Enter close", True, (230, 230, 230))
                self.screen.blit(bar, (8, 8))
                if hud_text:
                    hud = self.font.render(hud_text, True, (200, 200, 200))
                    self.screen.blit(hud, (8, 28))

        def run_once(self, grid: List[List[str]], title: str = "") -> None:
            if title:
                pygame.display.set_caption(title)
            running = True
            while running:
                running = self._handle_input()
                self._draw(grid, hud_text=title)
                pygame.display.flip()
                self.clock.tick(60)
            # leave pygame initialized so it can be reused in the same session

else:
    # Stub to keep references safe if pygame is not available
    class PygameMapViewer:  # type: ignore
        def __init__(self, *_, **__): ...
        def run_once(self, *_a, **_k): ...


# Map / Expedition layer (keep map_layer.py next to this file)
from map_layer import WorldMap, Expedition, ExpeditionStatus


# ----------------------------- Core Types ------------------------------------

class Resource(str, Enum):
    OXYGEN = "oxygen"
    WATER = "water"
    FOOD = "food"
    METALS = "metals"
    POLYMERS = "polymers"
    FUEL = "fuel"


def pretty_resource(res: Resource) -> str:
    return {
        Resource.OXYGEN: "O₂",
        Resource.WATER: "H₂O",
        Resource.FOOD: "Food",
        Resource.METALS: "Metals",
        Resource.POLYMERS: "Polymers",
        Resource.FUEL: "Fuel",
    }[res]


@dataclass
class Blueprint:
    code: str
    display_name: str
    build_cost: Dict[Resource, int]
    power_output: float = 0.0           # kW produced
    power_cost: float = 0.0             # kW consumed when operating
    outputs: Dict[Resource, float] = field(default_factory=dict)  # per sol
    inputs: Dict[Resource, float] = field(default_factory=dict)   # per sol
    workers_required: int = 0
    capacity_bonus: int = 0             # increases colonist capacity
    notes: str = ""


@dataclass
class Building:
    blueprint: Blueprint
    id: int
    nickname: Optional[str] = None
    condition: float = 100.0            # 0..100
    online: bool = True
    assigned_workers: int = 0           # 0..workers_required

    def to_dict(self) -> dict:
        return {
            "blueprint_code": self.blueprint.code,
            "id": self.id,
            "nickname": self.nickname,
            "condition": self.condition,
            "online": self.online,
            "assigned_workers": self.assigned_workers,
        }


@dataclass
class Environment:
    sol: int = 1
    solar_multiplier: float = 1.0
    duststorm_days: int = 0

    def tick(self):
        self.sol += 1
        if self.duststorm_days > 0:
            self.duststorm_days -= 1
            if self.duststorm_days == 0:
                self.solar_multiplier = 1.0


@dataclass
class Colony:
    name: str
    population: int
    capacity: int
    morale: float = 60.0     # 0..100
    health: float = 85.0     # 0..100 (avg)
    inventory: Dict[Resource, float] = field(default_factory=dict)
    buildings: List[Building] = field(default_factory=list)
    rng_seed: int = 42
    life_support_kw_per_colonist: float = 0.5

    # Runtime only (not serialized)
    _rng: random.Random = field(default_factory=random.Random, repr=False, compare=False)

    # Colonists reserved for expeditions (not available for auto-assignment)
    reserved_for_expedition: int = 0

    def __post_init__(self):
        self._rng.seed(self.rng_seed)
        # Ensure all resources appear in inventory
        for r in Resource:
            self.inventory.setdefault(r, 0.0)

    # ------------------------- Power / Workers / Prod -------------------------

    def total_capacity_bonus(self) -> int:
        return sum(b.blueprint.capacity_bonus for b in self.buildings)

    def power_generated(self, env: Environment) -> float:
        total = 0.0
        for b in self.buildings:
            if not b.online:
                continue
            out = b.blueprint.power_output
            # Solar arrays are affected by environment
            if out > 0 and "solar" in b.blueprint.code:
                out *= env.solar_multiplier
            # Condition degrades generation slightly
            out *= max(0.0, b.condition / 100.0)
            total += out
        return total

    def power_required(self) -> float:
        # Base life support power is non-negotiable
        ls = self.population * self.life_support_kw_per_colonist
        # Operational power: only for buildings that have workers assigned and are online
        ops = 0.0
        for b in self.buildings:
            if not b.online:
                continue
            if b.blueprint.power_cost <= 0:
                continue
            if b.blueprint.workers_required == 0:
                # Facilities without worker requirement draw power while online
                ops += b.blueprint.power_cost * (b.condition / 100.0)
            else:
                # Scale by worker utilization
                util = 0.0 if b.blueprint.workers_required == 0 else min(
                    1.0, b.assigned_workers / b.blueprint.workers_required
                )
                ops += b.blueprint.power_cost * util * (b.condition / 100.0)
        return ls + ops

    def worker_pool(self) -> int:
        # All colonists can work for now. You can add jobs/roles later.
        used = sum(min(b.assigned_workers, b.blueprint.workers_required) for b in self.buildings)
        return max(0, self.population - self.reserved_for_expedition - used)

    def auto_assign_workers(self):
        """
        Distribute workers toward life-support-critical production first.
        Priority order: water -> oxygen -> food -> metals -> everything else.
        """
        priorities = ["water_extractor", "electrolyzer", "greenhouse", "mine"]
        # Respect expedition reservations so we don't over-assign
        available = max(0, self.population - self.reserved_for_expedition)
        # Reset
        for b in self.buildings:
            b.assigned_workers = 0
        # Place workers by priority
        for code in priorities:
            for b in self.buildings:
                if b.blueprint.code != code or not b.online:
                    continue
                need = max(0, b.blueprint.workers_required - b.assigned_workers)
                to_assign = min(need, available)
                b.assigned_workers += to_assign
                available -= to_assign
                if available <= 0:
                    return
        # Fill remaining buildings in declaration order
        if available > 0:
            for b in self.buildings:
                if not b.online or b.blueprint.workers_required == 0:
                    continue
                need = max(0, b.blueprint.workers_required - b.assigned_workers)
                to_assign = min(need, available)
                b.assigned_workers += to_assign
                available -= to_assign
                if available <= 0:
                    return

    def _consume_inputs_for(self, b: Building, factor: float) -> float:
        """
        Attempt to consume inputs needed for a building scaled by factor (0..1),
        returning the actual feasible factor after input constraints.
        """
        if not b.blueprint.inputs or factor <= 0:
            return factor
        # Determine limiting ratio from available inventory
        required = {r: amt * factor for r, amt in b.blueprint.inputs.items()}
        ratios = []
        for r, amt in required.items():
            if amt <= 0:
                continue
            have = self.inventory.get(r, 0.0)
            ratios.append(have / amt if amt > 0 else 1.0)
        input_ratio = min(ratios) if ratios else 1.0
        input_ratio = max(0.0, min(1.0, input_ratio))
        # Consume inputs at the feasible rate
        for r, amt in b.blueprint.inputs.items():
            self.inventory[r] -= amt * factor * input_ratio
            if self.inventory[r] < 0:
                self.inventory[r] = 0.0  # safety clamp
        return factor * input_ratio

    def run_production(self, env: Environment, log: List[str]) -> Tuple[float, float, float]:
        """
        Execute a sol of production/consumption and return (gen_kW, need_kW, utilization_ratio).
        Production is scaled by: worker_utilization * power_utilization * condition * input_availability.
        """
        gen = self.power_generated(env)
        need = self.power_required()
        # Life support is hard-prioritized: compute remaining power for industry.
        life_support_kw = self.population * self.life_support_kw_per_colonist
        if gen < life_support_kw:
            # Blackout beyond life-support; production stalls completely.
            power_ratio = 0.0
            remaining_for_ops = 0.0
        else:
            remaining_for_ops = gen - life_support_kw
            ops_need = max(0.0, need - life_support_kw)
            power_ratio = 1.0 if ops_need <= 0 else max(0.0, min(1.0, remaining_for_ops / ops_need))

        # Run each building
        for b in self.buildings:
            if not b.online:
                continue

            # Facility condition scales effectiveness
            condition_scale = max(0.0, b.condition / 100.0)

            # Worker utilization
            util = 1.0
            if b.blueprint.workers_required > 0:
                util = min(1.0, b.assigned_workers / max(1, b.blueprint.workers_required))

            # Net scale before inputs
            scale = util * condition_scale
            # Buildings that only produce power already accounted in gen; skip I/O
            if b.blueprint.outputs or b.blueprint.inputs:
                # Apply power ratio to production facilities
                scale *= power_ratio
                # Try to consume inputs for feasible scale
                scale = self._consume_inputs_for(b, scale)
                # Emit outputs
                for r, amt in b.blueprint.outputs.items():
                    self.inventory[r] += amt * scale

            # Passive condition decay
            if b.blueprint.workers_required > 0 or b.blueprint.power_cost > 0:
                b.condition = max(5.0, b.condition - (0.03 + 0.02 * util))

        return gen, need, power_ratio

    # ----------------------------- Life Support -------------------------------

    def life_support(self, log: List[str]):
        """
        Consume daily life support resources per colonist.
        Shortfalls reduce health and morale.
        """
        per_colonist = {
            Resource.OXYGEN: 1.0,
            Resource.WATER: 1.0,
            Resource.FOOD: 1.0,
        }
        deficits = []
        for r, amt in per_colonist.items():
            need = amt * self.population
            have = self.inventory.get(r, 0.0)
            take = min(have, need)
            self.inventory[r] = have - take
            if take < need - 1e-9:
                deficits.append(r)

        if deficits:
            # Harsh penalties for deficits; more severe for oxygen/water than food.
            penalty = 0.0
            if Resource.OXYGEN in deficits:
                penalty += 18.0
            if Resource.WATER in deficits:
                penalty += 12.0
            if Resource.FOOD in deficits:
                penalty += 8.0
            self.health = max(0.0, self.health - penalty)
            self.morale = max(0.0, self.morale - (6.0 + 2.0 * len(deficits)))
            log.append(f"⚠️ Life support deficit: {', '.join(pretty_resource(d) for d in deficits)}")
        else:
            # Slow recovery when fully supplied
            self.health = min(100.0, self.health + 2.0)
            self.morale = min(100.0, self.morale + 1.0)

        # Population risk if health collapses
        if self.health <= 0.0 and self.population > 0:
            deaths = max(1, math.ceil(self.population * 0.1))
            self.population = max(0, self.population - deaths)
            self.health = 25.0
            self.morale = max(0.0, self.morale - 15.0)
            log.append(f"💀 {deaths} colonist(s) died due to critical conditions.")

    # ------------------------------ Building Ops ------------------------------

    def can_afford(self, cost: Dict[Resource, int]) -> bool:
        return all(self.inventory.get(r, 0.0) >= amt for r, amt in cost.items())

    def pay(self, cost: Dict[Resource, int]):
        for r, amt in cost.items():
            self.inventory[r] -= amt

    def build(self, bp: Blueprint, next_id: int, log: List[str]) -> Optional[Building]:
        if not self.can_afford(bp.build_cost):
            log.append("Not enough resources to build that.")
            return None
        self.pay(bp.build_cost)
        b = Building(blueprint=bp, id=next_id)
        self.buildings.append(b)
        self.capacity += bp.capacity_bonus
        log.append(f"🏗  Built: {bp.display_name}.")
        return b

    # ------------------------------- Serialization ----------------------------

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "population": self.population,
            "capacity": self.capacity,
            "morale": self.morale,
            "health": self.health,
            "inventory": {r.value: v for r, v in self.inventory.items()},
            "buildings": [b.to_dict() for b in self.buildings],
            "rng_seed": self.rng_seed,
            "life_support_kw_per_colonist": self.life_support_kw_per_colonist,
            "reserved_for_expedition": self.reserved_for_expedition,
        }

    @classmethod
    def from_dict(cls, data: dict, blueprints: Dict[str, Blueprint]) -> "Colony":
        inv = {Resource(k): float(v) for k, v in data["inventory"].items()}
        colony = cls(
            name=data["name"],
            population=int(data["population"]),
            capacity=int(data["capacity"]),
            morale=float(data["morale"]),
            health=float(data["health"]),
            inventory=inv,
            rng_seed=int(data.get("rng_seed", 42)),
            life_support_kw_per_colonist=float(data.get("life_support_kw_per_colonist", 0.5)),
        )
        buildings = []
        for bd in data["buildings"]:
            code = bd["blueprint_code"]
            bp = blueprints[code]
            b = Building(
                blueprint=bp,
                id=int(bd["id"]),
                nickname=bd.get("nickname"),
                condition=float(bd["condition"]),
                online=bool(bd["online"]),
                assigned_workers=int(bd["assigned_workers"]),
            )
            buildings.append(b)
        colony.buildings = buildings
        colony.capacity += colony.total_capacity_bonus()
        colony.__post_init__()
        colony.reserved_for_expedition = int(data.get("reserved_for_expedition", 0))
        return colony


# ------------------------------ Game Definition ------------------------------

def default_blueprints() -> Dict[str, Blueprint]:
    return {
        "solar_array": Blueprint(
            code="solar_array",
            display_name="Solar Array",
            build_cost={Resource.METALS: 10, Resource.POLYMERS: 2},
            power_output=10.0,
            notes="Provides power; output reduced during dust storms.",
        ),
        "habitat": Blueprint(
            code="habitat",
            display_name="Habitat Module",
            build_cost={Resource.METALS: 25, Resource.POLYMERS: 5},
            power_cost=2.0,
            capacity_bonus=4,
            notes="Increases living capacity.",
        ),
        "water_extractor": Blueprint(
            code="water_extractor",
            display_name="Water Extractor",
            build_cost={Resource.METALS: 20},
            power_cost=5.0,
            outputs={Resource.WATER: 5.0},
            workers_required=2,
            notes="Extracts ice/permafrost to produce water.",
        ),
        "electrolyzer": Blueprint(
            code="electrolyzer",
            display_name="Electrolyzer",
            build_cost={Resource.METALS: 15},
            power_cost=4.0,
            inputs={Resource.WATER: 4.0},
            outputs={Resource.OXYGEN: 4.0},
            workers_required=2,
            notes="Converts water into oxygen for breathing.",
        ),
        "greenhouse": Blueprint(
            code="greenhouse",
            display_name="Greenhouse",
            build_cost={Resource.METALS: 15, Resource.POLYMERS: 5},
            power_cost=6.0,
            inputs={Resource.WATER: 3.0},
            outputs={Resource.FOOD: 3.0, Resource.OXYGEN: 1.0},
            workers_required=3,
            notes="Grows food and adds a bit of oxygen.",
        ),
        "mine": Blueprint(
            code="mine",
            display_name="Regolith Mine",
            build_cost={Resource.METALS: 10},
            power_cost=6.0,
            outputs={Resource.METALS: 3.0},
            workers_required=3,
            notes="Extracts metals for construction.",
        ),
        "reactor": Blueprint(
            code="reactor",
            display_name="Compact Fission Reactor",
            build_cost={Resource.METALS: 60, Resource.POLYMERS: 20, Resource.FUEL: 5},
            power_output=30.0,
            notes="Reliable base-load power. Expensive to build.",
        ),
    }


def _world_to_char_grid(w: WorldMap) -> List[List[str]]:
    """
    Obtain a 2D character grid to render.
    We ask the WorldMap for an ASCII view large enough to cover the whole world,
    then convert the lines into a rectangular grid.
    """
    try:
        half_w, half_h = w.w // 2, w.h // 2  # attributes used elsewhere in this file
        radius = max(half_w, half_h)
    except Exception:
        radius = 10
    s = w.ascii_map(0, 0, view_radius=radius)
    lines = [ln.rstrip("\n") for ln in s.splitlines() if ln.strip() != ""]
    width = max((len(ln) for ln in lines), default=0)
    # Right-pad lines to equal width; keep all characters (including borders) so the
    # visual roughly matches the ASCII output.
    grid = [list(ln.ljust(width)) for ln in lines]
    return grid


@dataclass
class Game:
    colony: Colony
    env: Environment = field(default_factory=Environment)
    blueprints: Dict[str, Blueprint] = field(default_factory=default_blueprints)
    next_building_id: int = 1
    log: List[str] = field(default_factory=list)

    # Map / Expedition state
    world: Optional[WorldMap] = None
    expedition: Optional[Expedition] = None

    # Graphics toggle (set in new_game() based on pygame availability)
    use_pygame_map: bool = False

    # Non-serialized runtime viewer instance
    _pg_viewer: Optional[PygameMapViewer] = field(default=None, repr=False, compare=False)

    # --------------------------- Event System ---------------------------------

    def random_event(self):
        rnd = self.colony._rng.random()
        # ~20% chance per sol something happens
        if rnd > 0.2:
            return

        roll = self.colony._rng.random()
        if roll < 0.30:
            # Dust storm
            days = self.colony._rng.randint(2, 6)
            self.env.duststorm_days = days
            self.env.solar_multiplier = 0.35
            self.log.append(f"🌪️ Dust storm! Solar output reduced for {days} sol(s).")
        elif roll < 0.60 and self.colony.buildings:
            # Equipment failure
            b = self.colony._rng.choice(self.colony.buildings)
            dmg = self.colony._rng.uniform(8.0, 25.0)
            b.condition = max(5.0, b.condition - dmg)
            self.log.append(f"🛠️ Equipment failure in {b.blueprint.display_name} (-{dmg:.1f}% condition).")
        elif roll < 0.80 and self.colony.buildings:
            # Micro-meteor strike
            b = self.colony._rng.choice(self.colony.buildings)
            if self.colony._rng.random() < 0.25 and b.condition < 20.0:
                # Destroy
                self.colony.buildings.remove(b)
                self.log.append(f"☄️ Micro-meteor destroyed a {b.blueprint.display_name}!")
            else:
                dmg = self.colony._rng.uniform(15.0, 35.0)
                b.condition = max(5.0, b.condition - dmg)
                self.log.append(f"☄️ Micro-meteor damaged {b.blueprint.display_name} (-{dmg:.1f}%).")
        else:
            # Supply drop
            bonus = {
                Resource.WATER: self.colony._rng.randint(6, 14),
                Resource.FOOD: self.colony._rng.randint(4, 12),
                Resource.METALS: self.colony._rng.randint(8, 16),
                Resource.POLYMERS: self.colony._rng.randint(1, 4),
            }
            for r, v in bonus.items():
                self.colony.inventory[r] += v
            self.log.append("📦 Unscheduled supply pod landed nearby with mixed resources.")

    # --------------------------- Game Flow ------------------------------------

    def start_of_sol(self):
        self.log.clear()
        if self.colony.population > self.colony.capacity:
            overflow = self.colony.population - self.colony.capacity
            self.colony.morale = max(0.0, self.colony.morale - 2.0 * overflow)
            self.log.append(f"🏚 Overcapacity: {overflow} colonist(s) without proper habitat.")

    def end_of_sol(self):
        self.env.tick()

    def advance_one_sol(self):
        self.start_of_sol()
        # Auto-assign workers (simple baseline you can replace with a UI)
        self.colony.auto_assign_workers()
        gen, need, util = self.colony.run_production(self.env, self.log)
        self.colony.life_support(self.log)
        self.random_event()
        # Expeditions progress after core colony processing
        self._progress_expedition()
        self.end_of_sol()
        return gen, need, util

    # ------------------------- Exploration Layer ------------------------------

    def _ensure_world(self):
        if self.world is None:
            # world seeded from colony seed for deterministic layout
            self.world = WorldMap(21, 21, seed=self.colony.rng_seed)
            # reveal a small area around the base
            self.world.reveal(0, 0, radius=2)

    def _progress_expedition(self):
        if not self.expedition or not self.expedition.active:
            return
        logs, cargo, finished = self.expedition.tick(self.colony, self.env)
        self.log.extend(logs)
        if finished:
            # Deliver cargo and free workers
            for k, v in cargo.items():
                try:
                    res = Resource[k.upper()]
                    self.colony.inventory[res] += v
                except KeyError:
                    pass
            self.colony.reserved_for_expedition = max(
                0, self.colony.reserved_for_expedition - self.expedition.team_size
            )
            self.expedition = None

    def _show_map_graphics(self):
        """
        If pygame is available, render a tile view of the whole world. Otherwise, noop.
        """
        if not self.use_pygame_map or not _HAS_PYGAME:
            return
        try:
            grid = _world_to_char_grid(self.world)  # type: ignore[arg-type]
            if not grid or not grid[0]:
                return
            if self._pg_viewer is None:
                # Choose a tile size that tends to fit on a typical screen.
                # If the map is bigger, you can pan/zoom.
                tile_px = 24
                self._pg_viewer = PygameMapViewer(tile_px=tile_px, window_size=(1024, 768))
            title = f"Mars Map — Sol {self.env.sol}"
            self._pg_viewer.run_once(grid, title=title)
        except Exception as ex:
            # If something goes wrong (e.g., headless environment), fall back silently.
            self.log.append(f"(Map viewer unavailable: {ex})")

    def map_menu(self):
        self._ensure_world()
        w = self.world
        print("\n" + "=" * 64)
        print("Map / Expedition")
        # If graphics are enabled (and pygame is present), show the viewer; otherwise print ASCII.
        if self.use_pygame_map and _HAS_PYGAME:
            self._show_map_graphics()
        else:
            print(w.ascii_map(0, 0, view_radius=6))  # type: ignore[union-attr]

        if self.expedition and self.expedition.active:
            ex = self.expedition
            print(f"\nActive expedition: team={ex.team_size}, rover={'yes' if ex.use_rover else 'no'}, "
                  f"dest={ex.dest}, status={ex.status.value}, progress_step={ex.step_index}/{len(ex.path)}")
            input("Press Enter to return...")
            return

        # No active expedition -> allow launching one
        print("\nLaunch a new expedition from (0,0). Coordinates are relative to colony.")
        try:
            x = int(input("Destination X (e.g., 3): ").strip())
            y = int(input("Destination Y (e.g., -2): ").strip())
        except Exception:
            print("Invalid coordinates.")
            return

        # Bounds check
        half_w, half_h = w.w // 2, w.h // 2  # type: ignore[union-attr]
        if not (-half_w <= x <= half_w and -half_h <= y <= half_h):
            print("Destination outside world bounds.")
            return

        available_team_max = max(0, self.colony.population - self.colony.reserved_for_expedition)
        if available_team_max <= 0:
            print("No available colonists to send (all reserved/assigned).")
            return

        try:
            team = int(input(f"Team size (1..{available_team_max}): ").strip())
        except Exception:
            print("Invalid team size.")
            return
        if team < 1 or team > available_team_max:
            print("Team size out of range.")
            return

        use_rover = input("Use rover? (y/n) [y]: ").strip().lower() != "n"
        # Optional cost to use rover
        if use_rover:
            fuel_cost = 1
            if self.colony.inventory.get(Resource.FUEL, 0.0) < fuel_cost:
                print("Not enough Fuel to use rover (need 1). Launching on foot.")
                use_rover = False
            else:
                self.colony.inventory[Resource.FUEL] -= fuel_cost
                self.log.append("🛻 Rover prepped (-1 Fuel).")

        # Reserve workers and start
        self.colony.reserved_for_expedition += team
        self.expedition = Expedition(
            world=w, dest=(x, y), team_size=team, use_rover=use_rover,
            seed=self.colony._rng.randint(0, 10**9)
        )
        self.log.append(f"🧭 Expedition prepared to {x},{y} (team {team}, rover={'yes' if use_rover else 'no'}).")

    # --------------------------- Save / Load ----------------------------------

    def save(self, path: str = "savegame.json"):
        data = {
            "env": asdict(self.env),
            "colony": self.colony.to_dict(),
            "next_building_id": self.next_building_id,
            "world": self.world.to_dict() if self.world else None,
            "expedition": self.expedition.to_dict() if self.expedition else None,
            # Include whether gfx was enabled
            "use_pygame_map": self.use_pygame_map,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        self.log.append(f"💾 Saved to {path}")

    def load(self, path: str = "savegame.json"):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.env = Environment(**data["env"])
        self.colony = Colony.from_dict(data["colony"], self.blueprints)
        self.next_building_id = int(data.get("next_building_id", 1))
        # map / expedition (handle older saves gracefully)
        wdata = data.get("world")
        if wdata:
            self.world = WorldMap.from_dict(wdata)
        else:
            self.world = None
        edata = data.get("expedition")
        if edata and self.world:
            self.expedition = Expedition.from_dict(self.world, edata)
        else:
            self.expedition = None
        self.use_pygame_map = bool(data.get("use_pygame_map", self.use_pygame_map))
        self.log.append(f"📂 Loaded from {path}")

    # ------------------------------ UI Helpers --------------------------------

    def print_status(self):
        c = self.colony
        e = self.env
        print("\n" + "=" * 64)
        print(f"Sol {e.sol} — {c.name}")
        env_str = "Dust Storm" if e.duststorm_days > 0 else "Clear"
        print(f"Env: {env_str} (solar x{e.solar_multiplier:.2f})")
        print(f"Population: {c.population}/{c.capacity}  |  Morale: {c.morale:.0f}  |  Health: {c.health:.0f}")
        gen = c.power_generated(e)
        need = c.power_required()
        util = 1.0 if need == 0 else min(1.0, max(0.0, gen / max(1e-6, need)))
        print(f"Power: {gen:.1f} kW gen  /  {need:.1f} kW need  (util ~ {util*100:.0f}%)")
        # Inventory summary
        inv_parts = []
        for r in [Resource.OXYGEN, Resource.WATER, Resource.FOOD, Resource.METALS, Resource.POLYMERS, Resource.FUEL]:
            inv_parts.append(f"{pretty_resource(r)} {c.inventory.get(r, 0.0):.1f}")
        print("Inventory:", " | ".join(inv_parts))
        # Buildings
        if c.buildings:
            print("Buildings:")
            for b in c.buildings:
                util = 0.0
                if b.blueprint.workers_required > 0:
                    util = min(1.0, b.assigned_workers / max(1, b.blueprint.workers_required))
                tag = f"{b.blueprint.display_name} (cond {b.condition:.0f}%, workers {b.assigned_workers}/{b.blueprint.workers_required})"
                if "solar" in b.blueprint.code or b.blueprint.power_output > 0:
                    tag += f" [+{b.blueprint.power_output:.0f} kW]"
                if b.blueprint.power_cost > 0:
                    tag += f" [-{b.blueprint.power_cost:.0f} kW]"
                if not b.online:
                    tag += " [OFF]"
                print("  •", tag)
        else:
            print("Buildings: none")

        # Expedition quick status
        if self.expedition and self.expedition.active:
            ex = self.expedition
            print(f"\nExpedition: team={ex.team_size}, rover={'yes' if ex.use_rover else 'no'}, "
                  f"dest={ex.dest}, status={ex.status.value}, step {ex.step_index}/{len(ex.path)}")

        if self.log:
            print("\nEvents/Notes:")
            for line in self.log:
                print(" -", line)

    def choose_build(self):
        print("\nAvailable Blueprints:")
        keys = list(self.blueprints.keys())
        for i, key in enumerate(keys, 1):
            bp = self.blueprints[key]
            cost = ", ".join(f"{pretty_resource(r)} {amt}" for r, amt in bp.build_cost.items()) or "None"
            io = []
            if bp.outputs:
                io.append("out: " + ", ".join(f"{pretty_resource(r)} {v}/sol" for r, v in bp.outputs.items()))
            if bp.inputs:
                io.append("in: " + ", ".join(f"{pretty_resource(r)} {v}/sol" for r, v in bp.inputs.items()))
            power = []
            if bp.power_output > 0:
                power.append(f"+{bp.power_output:.0f}kW")
            if bp.power_cost > 0:
                power.append(f"-{bp.power_cost:.0f}kW")
            extras = []
            if bp.capacity_bonus:
                extras.append(f"+cap {bp.capacity_bonus}")
            summary = " | ".join(x for x in ["; ".join(io), " ".join(power), " ".join(extras)] if x)
            print(f" {i}. {bp.display_name:26}  Cost: {cost}  {summary}")
        print(" 0. Cancel")
        choice = input("Build which? ").strip()
        if not choice.isdigit():
            print("Invalid selection.")
            return
        ix = int(choice)
        if ix == 0:
            return
        if not (1 <= ix <= len(keys)):
            print("Invalid number.")
            return
        bp = self.blueprints[keys[ix - 1]]
        b = self.colony.build(bp, self.next_building_id, self.log)
        if b:
            self.next_building_id += 1

    def toggle_building(self):
        powered = [b for b in self.colony.buildings]
        if not powered:
            print("No buildings to toggle.")
            return
        print("\nToggle Building Online/Offline:")
        for i, b in enumerate(powered, 1):
            state = "ON " if b.online else "OFF"
            print(f" {i}. {b.blueprint.display_name} #{b.id} [{state}]")
        print(" 0. Cancel")
        raw = input("Select building: ").strip()
        if not raw.isdigit():
            print("Invalid selection.")
            return
        ix = int(raw)
        if ix == 0:
            return
        if not (1 <= ix <= len(powered)):
            print("Invalid number.")
            return
        b = powered[ix - 1]
        b.online = not b.online
        self.log.append(
            f"{'🔌 Disabled' if not b.online else '⚡ Enabled'} {b.blueprint.display_name} #{b.id}."
        )

    def explore(self):
        """
        A quick 'go exploring' action for flavor; small chance to find resources or trigger an event.
        """
        r = self.colony._rng.random()
        if r < 0.40:
            water = self.colony._rng.randint(2, 6)
            self.colony.inventory[Resource.WATER] += water
            self.log.append(f"🧭 Exploration: found subsurface ice (+H₂O {water}).")
        elif r < 0.70:
            metals = self.colony._rng.randint(2, 6)
            self.colony.inventory[Resource.METALS] += metals
            self.log.append(f"🧭 Exploration: located metal nodules (+Metals {metals}).")
        elif r < 0.85:
            self.colony.morale = min(100.0, self.colony.morale + 3.0)
            self.log.append("🧭 Exploration: scenic canyon vista lifted spirits (+Morale).")
        else:
            # Minor mishap
            self.colony.health = max(0.0, self.colony.health - 5.0)
            self.log.append("🧭 Exploration mishap: minor injuries (-Health).")

    # ------------------------------ Main Loop ---------------------------------

    def main_menu(self):
        while True:
            self.print_status()
            print("\nActions:")
            print(" 1) Build")
            print(" 2) Toggle building ON/OFF")
            print(" 3) Explore")
            print(" 4) Save game")
            print(" 5) Load game")
            print(" 6) End Sol")
            print(" 7) Map / Expedition")
            print(" 0) Quit")
            choice = input("Select action: ").strip()

            if choice == "1":
                self.choose_build()
            elif choice == "2":
                self.toggle_building()
            elif choice == "3":
                self.explore()
            elif choice == "4":
                self.save()
            elif choice == "5":
                try:
                    self.load()
                except FileNotFoundError:
                    self.log.append("No savegame.json found.")
            elif choice == "6":
                gen, need, util = self.advance_one_sol()
                # Auto feedback this sol
                self.log.append(f"End of Sol: power use {min(1.0, gen/max(need,1e-6))*100:.0f}% "
                                f"(gen {gen:.1f} kW / need {need:.1f} kW).")
                if self.colony.population <= 0:
                    self.print_status()
                    print("\nAll colonists lost. Simulation ends.")
                    return
            elif choice == "7":
                self.map_menu()
            elif choice == "0":
                print("Goodbye.")
                return
            else:
                print("Unknown option.")


# --------------------------- Game Initialization -----------------------------

def new_game(colony_name: str = "Ares Base", seed: int = 42, force_no_gfx: bool = False) -> Game:
    blueprints = default_blueprints()

    # Starting state (tune freely)
    starting_inventory = {
        Resource.OXYGEN: 50.0,
        Resource.WATER: 50.0,
        Resource.FOOD: 50.0,
        Resource.METALS: 50.0,
        Resource.POLYMERS: 10.0,
        Resource.FUEL: 2.0,
    }

    colony = Colony(
        name=colony_name,
        population=6,
        capacity=8,
        morale=65.0,
        health=90.0,
        inventory=starting_inventory,
        rng_seed=seed,
        life_support_kw_per_colonist=0.5,
    )

    # Prefab starting buildings
    starters = [
        "solar_array", "solar_array",
        "habitat",
        "water_extractor",
        "electrolyzer",
        "greenhouse",
    ]
    next_id = 1
    for code in starters:
        bp = blueprints[code]
        b = Building(blueprint=bp, id=next_id)
        next_id += 1
        colony.buildings.append(b)
        colony.capacity += bp.capacity_bonus

    game = Game(colony=colony, blueprints=blueprints, next_building_id=next_id)
    # Initialize world map with deterministic seed and reveal area around base
    game.world = WorldMap(21, 21, seed=seed)
    game.world.reveal(0, 0, radius=2)
    # Enable pygame map if available, unless explicitly disabled
    game.use_pygame_map = _HAS_PYGAME and not force_no_gfx
    return game


def main():
    # CLI flags:
    #   --load     : load savegame.json
    #   --no-gfx   : force terminal-only map (ignore pygame even if installed)
    force_no_gfx = "--no-gfx" in sys.argv
    if len(sys.argv) > 1 and sys.argv[1] == "--load":
        g = new_game(force_no_gfx=force_no_gfx)  # seed structure, will be overwritten by load
        try:
            g.load()
            print("Loaded savegame.json\n")
        except FileNotFoundError:
            print("No savegame.json found, starting new game.\n")
    else:
        g = new_game(force_no_gfx=force_no_gfx)

    g.main_menu()


if __name__ == "__main__":
    main()
