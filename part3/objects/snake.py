from collections import deque
from .base import GridObject


class Snake(GridObject):
    def __init__(self, grid_width, grid_height):
        super().__init__(grid_width, grid_height)
        self.reset()

    def reset(self):
        # 初始長度 3，位於地圖中間
        cx, cy = self.w // 2, self.h // 2
        self.body = deque([(cx, cy), (cx, cy+1), (cx, cy+2)])
        self.direction = (0, -1)  # 預設往上
        self.alive = True
        self.grow_pending = False

    @property
    def head(self):
        return self.body[0]

    def change_direction(self, action):
        """
        Action: 0=上, 1=下, 2=左, 3=右
        封裝規則：蛇不能直接 180 度迴轉
        """
        dirs = {0: (0, -1), 1: (0, 1), 2: (-1, 0), 3: (1, 0)}
        new_dir = dirs.get(action)
        if new_dir:
            if (new_dir[0] + self.direction[0] != 0) or (new_dir[1] + self.direction[1] != 0):
                self.direction = new_dir

    def move(self):
        if not self.alive:
            return

        head_x, head_y = self.body[0]
        dx, dy = self.direction
        new_head = (head_x + dx, head_y + dy)

        if not (0 <= new_head[0] < self.w and 0 <= new_head[1] < self.h):
            self.alive = False
            return

        if new_head in self.body:
            if new_head != self.body[-1]:
                self.alive = False
                return

        self.body.appendleft(new_head)
        if self.grow_pending:
            self.grow_pending = False
        else:
            self.body.pop()

    def grow(self):
        self.grow_pending = True

    def check_collision(self, pos):
        return pos in self.body
