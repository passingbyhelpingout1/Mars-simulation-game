# renderers/pygame_renderer.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Sequence, Tuple
import pygame

Vec2 = Tuple[int, int]

def slice_tileset(atlas_path: Path, tile_size: int) -> List[pygame.Surface]:
    img = pygame.image.load(str(atlas_path)).convert_alpha()
    w, h = img.get_width(), img.get_height()
    tiles: List[pygame.Surface] = []
    for ty in range(0, h, tile_size):
        for tx in range(0, w, tile_size):
            tile = pygame.Surface((tile_size, tile_size), pygame.SRCALPHA)
            tile.blit(img, (0, 0), pygame.Rect(tx, ty, tile_size, tile_size))
            tiles.append(tile)
    return tiles

class PygameRenderer:
    def __init__(
        self,
        atlas: Path,
        tile_size: int = 32,
        char_to_index: Dict[str, int] | None = None,
        window_size: Vec2 = (960, 600),
        title: str = "Mars Colony"
    ):
        pygame.init()
        self.screen = pygame.display.set_mode(window_size)
        pygame.display.set_caption(title)
        self.clock = pygame.time.Clock()
        self.tile_size = tile_size
        self.tiles = slice_tileset(atlas, tile_size)
        self.char_to_index = char_to_index or {}
        self.cam_x = 0
        self.cam_y = 0
        self.zoom = 1.0

    def _handle_input(self) -> bool:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                return False
            if e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE:
                return False

        keys = pygame.key.get_pressed()
        px = int(400 * self.clock.get_time() / 1000)  # pan speed in px/s
        if keys[pygame.K_LEFT]:  self.cam_x -= px
        if keys[pygame.K_RIGHT]: self.cam_x += px
        if keys[pygame.K_UP]:    self.cam_y -= px
        if keys[pygame.K_DOWN]:  self.cam_y += px
        if keys[pygame.K_MINUS] or keys[pygame.K_KP_MINUS]:
            self.zoom = max(0.5, self.zoom - 0.05)
        if keys[pygame.K_EQUALS] or keys[pygame.K_KP_PLUS]:
            self.zoom = min(3.0, self.zoom + 0.05)
        return True

    def _draw_grid(self, grid: Sequence[Sequence[str]]) -> None:
        ts = max(1, int(self.tile_size * self.zoom))
        self.screen.fill((12, 10, 10))

        h = len(grid)
        w = len(grid[0]) if h else 0

        # Compute visible tile rectangle
        view_x = max(0, self.cam_x // ts)
        view_y = max(0, self.cam_y // ts)
        tiles_x = self.screen.get_width() // ts + 2
        tiles_y = self.screen.get_height() // ts + 2

        start_sx = - (self.cam_x % ts)
        sy = - (self.cam_y % ts)
        for gy in range(view_y, min(view_y + tiles_y, h)):
            sx = start_sx
            for gx in range(view_x, min(view_x + tiles_x, w)):
                ch = grid[gy][gx]
                idx = self.char_to_index.get(ch)
                if idx is not None and 0 <= idx < len(self.tiles):
                    tile = self.tiles[idx]
                    if ts != self.tile_size:
                        tile = pygame.transform.scale(tile, (ts, ts))
                    self.screen.blit(tile, (sx, sy))
                else:
                    # Fallback block if a character isn't mapped
                    pygame.draw.rect(self.screen, (40, 40, 44), (sx, sy, ts, ts))
                sx += ts
            sy += ts

    def draw_once(self, grid: Sequence[Sequence[str]]) -> bool:
        if not self._handle_input():
            return False
        self._draw_grid(grid)
        pygame.display.flip()
        self.clock.tick(60)      # cap at ~60 FPS
        return True
