import random
from collections import deque
from abc import ABC, abstractmethod


class GridObject(ABC):
    """Base class for anything that lives on the discrete grid."""

    def __init__(self, grid_width, grid_height):
        self.w = grid_width
        self.h = grid_height

    @abstractmethod
    def reset(self):
        """Subclasses provide their own reset behavior."""
        raise NotImplementedError

class Snake(GridObject):
    def __init__(self, grid_width, grid_height):
        super().__init__(grid_width, grid_height)
        self.reset()

    def reset(self):
        # 初始長度 3，位於地圖中間
        cx, cy = self.w // 2, self.h // 2
        # 使用 deque (雙向佇列) 管理身體，效率比 list 高
        # body[0] 是頭
        self.body = deque([(cx, cy), (cx, cy+1), (cx, cy+2)])
        self.direction = (0, -1) # 預設往上
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
        # 定義方向向量 (dx, dy)
        dirs = {0: (0, -1), 1: (0, 1), 2: (-1, 0), 3: (1, 0)}
        new_dir = dirs.get(action)

        if new_dir:
            # 防止反向移動 (例如正在往上，不能直接往下)
            if (new_dir[0] + self.direction[0] != 0) or (new_dir[1] + self.direction[1] != 0):
                self.direction = new_dir

    def move(self):
        """計算下一步的位置"""
        if not self.alive:
            return

        head_x, head_y = self.body[0]
        dx, dy = self.direction
        new_head = (head_x + dx, head_y + dy)

        # 1. 撞牆判斷
        if not (0 <= new_head[0] < self.w and 0 <= new_head[1] < self.h):
            self.alive = False
            return

        # 2. 撞身體判斷 (咬到自己)
        if new_head in self.body:
            # 特例：如果尾巴剛好移開，就不算撞到，但這裡簡化處理
            if new_head != self.body[-1]: 
                self.alive = False
                return

        # 3. 移動邏輯
        self.body.appendleft(new_head) # 加頭
        
        if self.grow_pending:
            self.grow_pending = False # 變長，不移除尾巴
        else:
            self.body.pop() # 沒吃到東西，移除尾巴

    def grow(self):
        """外部呼叫此方法讓蛇變長"""
        self.grow_pending = True

    def check_collision(self, pos):
        """檢查某個座標是否在蛇身上 (給 Food 用)"""
        return pos in self.body


class Food(GridObject):
    def __init__(self, grid_width, grid_height):
        super().__init__(grid_width, grid_height)
        self.position = (0, 0)

    def reset(self):
        # 食物的 reset 讓呼叫者決定位置，這裡設為原點以防未初始化
        self.position = (0, 0)
    
    def respawn(self, snake_body):
        """
        隨機生成食物，確保不會生成在蛇身上
        """
        while True:
            x = random.randint(0, self.w - 1)
            y = random.randint(0, self.h - 1)
            if (x, y) not in snake_body:
                self.position = (x, y)
                break