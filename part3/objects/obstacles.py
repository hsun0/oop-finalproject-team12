import random
from .base import GridObject


class Obstacle(GridObject):

    def __init__(self, grid_width, grid_height, max_obstacles=40, spawn_chance=0.1, despawn_chance=0.02):
        super().__init__(grid_width, grid_height)
        self.max_obstacles = max_obstacles
        self.spawn_chance = spawn_chance
        self.despawn_chance = despawn_chance
        self.positions = set()
        self.objects: list[dict] = []
        self.shape_probs = {
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
        if self.objects:
            self._evolve_all()

        if self.objects and random.random() < self.despawn_chance:
            idx = random.randrange(0, len(self.objects))
            removed = self.objects.pop(idx)
            for cell in removed["cells"]:
                if cell in self.positions:
                    self.positions.remove(cell)

        if len(self.positions) >= self.max_obstacles:
            return

        if random.random() < self.spawn_chance:
            blocked = set(snake_body) | self.positions | {tuple(food_pos)}
            shape_cls = self._sample_shape_class()
            shape_obj = shape_cls(self.w, self.h)
            for _ in range(50):
                x = random.randint(0, self.w - 1)
                y = random.randint(0, self.h - 1)
                cells = shape_obj.shape_cells(x, y)
                if not cells:
                    continue
                if all((0 <= cx < self.w and 0 <= cy < self.h and (cx, cy) not in blocked) for (cx, cy) in cells):
                    obj = {"cells": cells, "color": getattr(shape_obj, "color", (120, 120, 120))}
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

    def shape_cells(self, x: int, y: int):
        return [(x, y)]

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
            if not cells:
                continue

            candidates = set(cells)
            for c in list(cells):
                candidates.update(self._neighbors8(c))

            next_cells = set()
            for c in candidates:
                x, y = c
                if not (0 <= x < self.w and 0 <= y < self.h):
                    continue
                live_neighbors = sum((nbr in cells) for nbr in self._neighbors8(c))
                if c in cells:
                    if live_neighbors in (2, 3):
                        next_cells.add(c)
                else:
                    if live_neighbors == 3:
                        next_cells.add(c)

            filtered = []
            for c in next_cells:
                if c not in new_positions:
                    filtered.append(c)
                    new_positions.add(c)

            if filtered:
                new_objects.append({"cells": filtered, "color": obj.get("color", (120, 120, 120))})

        self.objects = new_objects
        self.positions = new_positions


class SingleObstacle(Obstacle):
    def shape_cells(self, x: int, y: int):
        return [(x, y)]
    color = (160, 160, 160)


class LShapeObstacle(Obstacle):
    def shape_cells(self, x: int, y: int):
        return [(x, y), (x, y+1), (x+1, y)]
    color = (100, 149, 237)


class CubeObstacle(Obstacle):
    def shape_cells(self, x: int, y: int):
        return [(x, y), (x+1, y), (x, y+1), (x+1, y+1)]
    color = (200, 200, 60)


class BeehiveObstacle(Obstacle):
    def shape_cells(self, x: int, y: int):
        return [(x+1, y+0), (x+2, y+0), (x+0, y+1), (x+3, y+1), (x+1, y+2), (x+2, y+2)]
    color = (255, 165, 0)


class LoafObstacle(Obstacle):
    def shape_cells(self, x: int, y: int):
        return [(x+1, y+0), (x+2, y+0), (x+0, y+1), (x+3, y+1), (x+1, y+2), (x+3, y+2), (x+2, y+3)]
    color = (34, 139, 34)


class BoatObstacle(Obstacle):
    def shape_cells(self, x: int, y: int):
        return [(x+0, y+0), (x+1, y+0), (x+0, y+1), (x+2, y+1), (x+1, y+2)]
    color = (70, 130, 180)


class TubObstacle(Obstacle):
    def shape_cells(self, x: int, y: int):
        return [(x+1, y+0), (x+0, y+1), (x+2, y+1), (x+1, y+2)]
    color = (147, 112, 219)


class BlinkerObstacle(Obstacle):
    def shape_cells(self, x: int, y: int):
        return [(x+0, y+0), (x+1, y+0), (x+2, y+0)]
    color = (255, 105, 180)


class ToadObstacle(Obstacle):
    def shape_cells(self, x: int, y: int):
        return [(x+1, y+0), (x+2, y+0), (x+3, y+0), (x+0, y+1), (x+1, y+1), (x+2, y+1)]
    color = (0, 206, 209)


class BeaconObstacle(Obstacle):
    def shape_cells(self, x: int, y: int):
        return [(x+0, y+0), (x+1, y+0), (x+0, y+1), (x+1, y+1), (x+2, y+2), (x+3, y+2), (x+2, y+3), (x+3, y+3)]
    color = (255, 140, 0)


class GliderObstacle(Obstacle):
    def shape_cells(self, x: int, y: int):
        return [(x+1, y+0), (x+2, y+1), (x+0, y+2), (x+1, y+2), (x+2, y+2)]
    color = (0, 255, 127)
