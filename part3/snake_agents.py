import random
from collections import deque
from abc import ABC, abstractmethod

class BaseAgent(ABC):
    def __init__(self, name):
        self.name = name

    @abstractmethod
    def select_action(self, obs):
        pass

class RandomAgent(BaseAgent):
    def __init__(self):
        super().__init__("Random Agent")

    def select_action(self, obs):
        # 隨機回傳 0~3
        return random.randint(0, 3)

class GreedyAgent(BaseAgent):
    def __init__(self):
        super().__init__("Greedy Agent")

    def select_action(self, obs):
        hx, hy = obs['head']
        fx, fy = obs['food']

        # 簡單邏輯：先縮短 X 軸距離，再縮短 Y 軸
        # 0=Up, 1=Down, 2=Left, 3=Right
        if hx > fx: return 2 # 往左
        if hx < fx: return 3 # 往右
        if hy > fy: return 0 # 往上
        if hy < fy: return 1 # 往下
        
        return random.randint(0, 3) # 如果重疊(理論上不會)就隨機

class RuleBasedAgent(BaseAgent):
    def __init__(self):
        super().__init__("Safe Rule Agent")

    def select_action(self, obs):
        head = obs['head']
        food = obs['food']
        body = obs['body']
        w, h = obs['grid_size']

        possible_moves = [0, 1, 2, 3] # Up, Down, Left, Right
        safe_moves = []

        # 1. 篩選出不會立刻撞牆或撞身體的動作
        for action in possible_moves:
            nx, ny = head
            if action == 0: ny -= 1
            elif action == 1: ny += 1
            elif action == 2: nx -= 1
            elif action == 3: nx += 1

            # 檢查邊界
            if not (0 <= nx < w and 0 <= ny < h):
                continue
            # 檢查身體 (只要不在身體list裡就是安全的)
            if (nx, ny) in body:
                continue
                
            safe_moves.append(action)

        # 如果無路可走，只好等死 (或隨機選一個)
        if not safe_moves:
            return random.randint(0, 3)

        # 2. 在安全動作中，選擇離食物最近的 (貪婪策略)
        best_action = safe_moves[0]
        min_dist = float('inf')

        for action in safe_moves:
            nx, ny = head
            if action == 0: ny -= 1
            elif action == 1: ny += 1
            elif action == 2: nx -= 1
            elif action == 3: nx += 1
            
            # 計算曼哈頓距離
            dist = abs(nx - food[0]) + abs(ny - food[1])
            
            if dist < min_dist:
                min_dist = dist
                best_action = action
        
        return best_action


class PathfindingAgent(BaseAgent):
    def __init__(self):
        super().__init__("Pathfinding Agent")

    def select_action(self, obs):
        head = tuple(obs["head"])
        food = tuple(obs["food"])
        body = [tuple(p) for p in obs["body"]]
        w, h = obs["grid_size"]

        # 將尾巴暫時視為可通行，因為下一步尾巴會移動 (簡化假設)
        blocked = set(body[:-1]) if len(body) > 1 else set()

        path = self._bfs(head, food, blocked, w, h)
        if len(path) >= 2:
            return self._direction_from_step(head, path[1])

        # 若找不到路徑，退化為安全隨機行動
        return self._safe_fallback(head, blocked, w, h)

    def _bfs(self, start, goal, blocked, w, h):
        queue = deque([(start, [start])])
        visited = {start}
        while queue:
            pos, path = queue.popleft()
            if pos == goal:
                return path
            for nx, ny in self._neighbors(pos, w, h):
                if (nx, ny) in blocked or (nx, ny) in visited:
                    continue
                visited.add((nx, ny))
                queue.append(((nx, ny), path + [(nx, ny)]))
        return []

    def _neighbors(self, pos, w, h):
        x, y = pos
        for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h:
                yield nx, ny

    def _direction_from_step(self, head, target):
        hx, hy = head
        tx, ty = target
        if ty < hy:
            return 0  # Up
        if ty > hy:
            return 1  # Down
        if tx < hx:
            return 2  # Left
        return 3      # Right

    def _safe_fallback(self, head, blocked, w, h):
        safe_moves = []
        for action, (dx, dy) in enumerate([(0, -1), (0, 1), (-1, 0), (1, 0)]):
            nx, ny = head[0] + dx, head[1] + dy
            if 0 <= nx < w and 0 <= ny < h and (nx, ny) not in blocked:
                safe_moves.append(action)
        if safe_moves:
            return random.choice(safe_moves)
        return random.randint(0, 3)