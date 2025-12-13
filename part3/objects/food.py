import random
from .base import GridObject


class Food(GridObject):
    def __init__(self, grid_width, grid_height):
        super().__init__(grid_width, grid_height)
        self.position = (0, 0)
        self.value = 1  # 預設分數
        self.color = (255, 0, 0)  # 預設紅色

    def reset(self):
        self.position = (0, 0)

    def respawn(self, snake_body, forbidden_positions=None):
        blocked = set(snake_body)
        if forbidden_positions:
            blocked |= {tuple(p) for p in forbidden_positions}

        while True:
            x = random.randint(0, self.w - 1)
            y = random.randint(0, self.h - 1)
            if (x, y) not in blocked:
                self.position = (x, y)
                break


class RedFood(Food):
    def __init__(self, grid_width, grid_height):
        super().__init__(grid_width, grid_height)
        self.value = 1
        self.color = (255, 0, 0)  # 紅色


class GoldFood(Food):
    def __init__(self, grid_width, grid_height):
        super().__init__(grid_width, grid_height)
        self.value = 2
        self.color = (255, 215, 0)  # 金色
