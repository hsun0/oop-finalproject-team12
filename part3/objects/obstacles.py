import random
from .base import GridObject
from .obstacle_shape import *
from .evolution import EvolutionPolicy, ConwayEvolutionPolicy, NoEvolutionPolicy


class Obstacle(GridObject):
    """管理者：負責生成/移除/演化，並暴露既有 API（positions, update, get_colored_cells）。"""

    def __init__(self, grid_width, grid_height, max_obstacles=40, spawn_chance=0.1, despawn_chance=0.02, evolution_policy: EvolutionPolicy | None = None):
        super().__init__(grid_width, grid_height)
        self.max_obstacles = max_obstacles
        self.spawn_chance = spawn_chance
        self.despawn_chance = despawn_chance
        self.positions = set()
        # objects: list of {cells: list[(x,y)], color: (r,g,b), static: bool}
        self.objects: list[dict] = []
        self.evolution_policy: EvolutionPolicy = evolution_policy or ConwayEvolutionPolicy()
        # 形狀生成策略（加權隨機）
        self.shape_probs: dict[type[ObstacleShape], float] = {
            CubeObstacle: 0.1,
            LShapeObstacle: 0.1,
            BeehiveObstacle: 0.1,
            LoafObstacle: 0.1,
            BoatObstacle: 0.1,
            TubObstacle: 0.1,
            BlinkerObstacle: 0.12,
            ToadObstacle: 0.12,
            BeaconObstacle: 0.11,
            GliderObstacle: 0.05,
        }

    def reset(self):
        self.positions.clear()
        self.objects.clear()

    def update(self, snake_body, food_pos):
        # 生命遊戲演化（跳過 static 物件）
        if self.objects:
            self._evolve_all()

        # 隨機整體移除一個物件
        if self.objects and random.random() < self.despawn_chance:
            idx = random.randrange(0, len(self.objects))
            removed = self.objects.pop(idx)
            for cell in removed["cells"]:
                self.positions.discard(cell)

        # 容量限制（以總佔用格數）
        if len(self.positions) >= self.max_obstacles:
            return

        # 生成新物件
        if random.random() < self.spawn_chance:
            blocked = set(snake_body) | self.positions | {tuple(food_pos)}
            shape_cls = self._sample_shape_class()
            shape_obj: ObstacleShape = shape_cls()
            for _ in range(50):
                x = random.randint(0, self.w - 1)
                y = random.randint(0, self.h - 1)
                cells = shape_obj.shape_cells(x, y)
                if not cells:
                    continue
                if all((0 <= cx < self.w and 0 <= cy < self.h and (cx, cy) not in blocked) for (cx, cy) in cells):
                    obj = {"cells": cells, "color": getattr(shape_obj, "color", (120, 120, 120)), "static": getattr(shape_obj, "static", False)}
                    self.objects.append(obj)
                    for cell in cells:
                        self.positions.add(cell)
                    break

    def check_collision(self, pos):
        return tuple(pos) in self.positions

    def get_colored_cells(self):
        colored = []
        for obj in self.objects:
            col = obj.get("color", (120, 120, 120))
            for cell in obj.get("cells", []):
                colored.append((cell, col))
        return colored

    def _sample_shape_class(self):
        r = random.random()
        acc = 0.0
        for cls, p in self.shape_probs.items():
            acc += p
            if r < acc:
                return cls
        return SingleObstacle

    @staticmethod
    def _neighbors8(cell):
        x, y = cell
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                yield (x + dx, y + dy)

    def _evolve_all(self):
        new_objects = []
        new_positions = set()
        for obj in self.objects:
            cells = set(obj.get("cells", []))
            if obj.get("static", False):
                next_cells = cells
            else:
                in_bounds = lambda c: (0 <= c[0] < self.w and 0 <= c[1] < self.h)
                next_cells = self.evolution_policy.evolve(cells, self._neighbors8, in_bounds)

            filtered = []
            for c in next_cells:
                if c not in new_positions and (0 <= c[0] < self.w and 0 <= c[1] < self.h):
                    filtered.append(c)
                    new_positions.add(c)

            if filtered:
                new_objects.append({"cells": filtered, "color": obj.get("color", (120, 120, 120)), "static": obj.get("static", False)})

        self.objects = new_objects
        self.positions = new_positions