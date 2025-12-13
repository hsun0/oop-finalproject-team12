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
        self.value = 1  # 預設分數
        self.color = (255, 0, 0)  # 預設紅色

    def reset(self):
        # 食物的 reset 讓呼叫者決定位置，這裡設為原點以防未初始化
        self.position = (0, 0)
    
    def respawn(self, snake_body, forbidden_positions=None):
        """隨機生成食物，確保不會生成在蛇身或障礙物上。"""
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


class Obstacle(GridObject):

    def __init__(self, grid_width, grid_height, max_obstacles=40, spawn_chance=0.1, despawn_chance=0.02):
        super().__init__(grid_width, grid_height)
        self.max_obstacles = max_obstacles
        self.spawn_chance = spawn_chance
        self.despawn_chance = despawn_chance
        # 目前所有障礙物的 cell 集合（快取）
        self.positions = set()
        # 以物件為單位管理；每個物件包含 cells 與 color
        self.objects: list[dict] = []
        # 生成型態的機率分佈：包含康威生命遊戲穩定態
        self.shape_probs = {
            CubeObstacle: 0.1,      # block（穩定態）
            LShapeObstacle: 0.1,    # L 形（穩定態）
            BeehiveObstacle: 0.1,   # 蜂巢（穩定態）
            LoafObstacle: 0.1,      # 麵包（穩定態）
            BoatObstacle: 0.1,      # 船（穩定態）
            TubObstacle: 0.1,       # 浴缸（穩定態）
            BlinkerObstacle: 0.12,   # 擺動子（會變化）
            ToadObstacle: 0.12,      # 擺動子（會變化）
            BeaconObstacle: 0.11,    # 擺動子（會變化）
            GliderObstacle: 0.05,    # 滑翔者（會移動）
        }

    def reset(self):
        self.positions.clear()
        self.objects.clear()

    def update(self, snake_body, food_pos):
        # 先讓既有障礙物依康威生命遊戲規則演化（每個物件獨立演化）
        if self.objects:
            self._evolve_all()

        # 隨機移除一整個障礙物（以物件為單位）
        if self.objects and random.random() < self.despawn_chance:
            idx = random.randrange(0, len(self.objects))
            removed = self.objects.pop(idx)
            for cell in removed["cells"]:
                if cell in self.positions:
                    self.positions.remove(cell)

        # 隨機新增障礙物：容量依據佔據的格子數量
        if len(self.positions) >= self.max_obstacles:
            return

        if random.random() < self.spawn_chance:
            blocked = set(snake_body) | self.positions | {tuple(food_pos)} # 避免生成在這些位置
            shape_cls = self._sample_shape_class()
            shape_obj = shape_cls(self.w, self.h)
            for _ in range(50):  # 最多嘗試幾次
                x = random.randint(0, self.w - 1)
                y = random.randint(0, self.h - 1)
                cells = shape_obj.shape_cells(x, y)
                if not cells:
                    continue
                # 確保所有格子合法且不碰撞
                if all((0 <= cx < self.w and 0 <= cy < self.h and (cx, cy) not in blocked) for (cx, cy) in cells):
                    # 加入為一個獨立障礙物（不可與既有重疊）
                    obj = {"cells": cells, "color": getattr(shape_obj, "color", (120, 120, 120))}
                    self.objects.append(obj)
                    for cell in cells:
                        self.positions.add(cell)
                    break

    def check_collision(self, pos):
        return tuple(pos) in self.positions

    def get_colored_cells(self):
        """展平成 [(cell, color), ...] 供 renderer 使用。"""
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

    # Base implementation (single cell); subclasses override
    def shape_cells(self, x: int, y: int):
        return [(x, y)]

    # --- Conway's Game of Life helpers ---
    @staticmethod
    def _neighbors8(cell):
        x, y = cell
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                yield (x + dx, y + dy)

    def _evolve_all(self):
        """讓每個障礙物依生命遊戲規則各自演化，並重建 positions。"""
        new_objects = []
        new_positions = set()
        for obj in self.objects:
            cells = set(obj.get("cells", []))
            if not cells:
                continue

            # 只考慮該物件自身的細胞作為活細胞，鄰居可能在其周圍誕生
            candidates = set(cells)
            for c in list(cells):
                candidates.update(self._neighbors8(c))

            next_cells = set()
            for c in candidates:
                x, y = c
                # 邊界外不計
                if not (0 <= x < self.w and 0 <= y < self.h):
                    continue
                live_neighbors = sum((nbr in cells) for nbr in self._neighbors8(c))
                if c in cells:
                    # 存活細胞: 2或3鄰居存活
                    if live_neighbors in (2, 3):
                        next_cells.add(c)
                else:
                    # 死細胞: 恰有3鄰居則誕生
                    if live_neighbors == 3:
                        # 不能與現有其他物件或整體 positions 重疊，稍後整體檢查；先暫存
                        next_cells.add(c)

            # 避免與其他物件重疊：先不過濾，稍後合併時用 new_positions 過濾
            filtered = []
            for c in next_cells:
                if c not in new_positions:
                    filtered.append(c)
                    new_positions.add(c)

            if filtered:
                new_objects.append({"cells": filtered, "color": obj.get("color", (120, 120, 120))})

        # 更新快取
        self.objects = new_objects
        self.positions = new_positions


class SingleObstacle(Obstacle):
    def shape_cells(self, x: int, y: int):
        return [(x, y)]
    color = (160, 160, 160)  # 淺灰


class LShapeObstacle(Obstacle):
    def shape_cells(self, x: int, y: int):
        return [(x, y), (x, y+1), (x+1, y)]
    color = (100, 149, 237)  # CornflowerBlue


class CubeObstacle(Obstacle):
    def shape_cells(self, x: int, y: int):
        return [(x, y), (x+1, y), (x, y+1), (x+1, y+1)]
    color = (200, 200, 60)  # 深紅


# === Conway's Game of Life stable patterns ===
class BeehiveObstacle(Obstacle):
    def shape_cells(self, x: int, y: int):
        # Beehive coordinates relative to (x,y) as top-left of bounding box
        # Pattern cells: (1,0),(2,0),(0,1),(3,1),(1,2),(2,2)
        return [(x+1, y+0), (x+2, y+0), (x+0, y+1), (x+3, y+1), (x+1, y+2), (x+2, y+2)]
    color = (255, 165, 0)  # Orange

class LoafObstacle(Obstacle):
    def shape_cells(self, x: int, y: int):
        # Loaf cells: (1,0),(2,0),(0,1),(3,1),(1,2),(3,2),(2,3)
        return [(x+1, y+0), (x+2, y+0), (x+0, y+1), (x+3, y+1), (x+1, y+2), (x+3, y+2), (x+2, y+3)]
    color = (34, 139, 34)  # ForestGreen

class BoatObstacle(Obstacle):
    def shape_cells(self, x: int, y: int):
        # Boat cells: (0,0),(1,0),(0,1),(2,1),(1,2)
        return [(x+0, y+0), (x+1, y+0), (x+0, y+1), (x+2, y+1), (x+1, y+2)]
    color = (70, 130, 180)  # SteelBlue

class TubObstacle(Obstacle):
    def shape_cells(self, x: int, y: int):
        # Tub cells: (1,0),(0,1),(2,1),(1,2)
        return [(x+1, y+0), (x+0, y+1), (x+2, y+1), (x+1, y+2)]
    color = (147, 112, 219)  # MediumPurple

# === Oscillators & Movers ===
class BlinkerObstacle(Obstacle):
    def shape_cells(self, x: int, y: int):
        # Horizontal blinker (period 2): cells at (x,y), (x+1,y), (x+2,y)
        return [(x+0, y+0), (x+1, y+0), (x+2, y+0)]
    color = (255, 105, 180)  # HotPink

class ToadObstacle(Obstacle):
    def shape_cells(self, x: int, y: int):
        # Toad (period 2), bounding box 4x2
        return [(x+1, y+0), (x+2, y+0), (x+3, y+0), (x+0, y+1), (x+1, y+1), (x+2, y+1)]
    color = (0, 206, 209)  # DarkTurquoise

class BeaconObstacle(Obstacle):
    def shape_cells(self, x: int, y: int):
        # Beacon (period 2), two 2x2 blocks separated by one cell
        return [(x+0, y+0), (x+1, y+0), (x+0, y+1), (x+1, y+1), (x+2, y+2), (x+3, y+2), (x+2, y+3), (x+3, y+3)]
    color = (255, 140, 0)  # DarkOrange

class GliderObstacle(Obstacle):
    def shape_cells(self, x: int, y: int):
        # Glider pattern (moves diagonally). Using classic orientation.
        return [(x+1, y+0), (x+2, y+1), (x+0, y+2), (x+1, y+2), (x+2, y+2)]
    color = (0, 255, 127)  # SpringGreen