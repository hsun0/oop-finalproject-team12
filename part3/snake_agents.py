import random
from collections import deque
from abc import ABC, abstractmethod

class BaseAgent(ABC):
    def __init__(self, name):
        self.name = name

    @abstractmethod
    def select_action(self, obs):
        pass

    def action_to_dir(self, action: int) -> tuple[int, int]:
        mapping = {
            0: (0, -1),  # Up
            1: (0, 1),   # Down
            2: (-1, 0),  # Left
            3: (1, 0),   # Right
        }
        return mapping.get(action, (0, 0))

    def dir_to_action(self, dir: tuple[int, int]) -> int:
        reverse = {
            (0, -1): 0,
            (0, 1): 1,
            (-1, 0): 2,
            (1, 0): 3,
        }
        return reverse.get(tuple(dir), random.randint(0, 3))
    
    def head_valid(self, obs, dir: tuple[int, int]) -> bool:
        head = obs['head']
        body = obs['body']
        w, h = obs['grid_size']
        x, y = head[0] + dir[0], head[1] + dir[1]
        if not (0 <= x < w and 0 <= y < h):
            return False
        if (x, y) in body:
            return False
        return True

    def neighbors(self, head: tuple[int, int]):
        """回傳四鄰居 (位置, 對應動作碼, 方向向量)"""
        for action in [0, 1, 2, 3]:
            dx, dy = self.action_to_dir(action)
            yield (head[0] + dx, head[1] + dy), action, (dx, dy)

    def safe_actions(self, obs):
        """回傳在當前觀察下的安全動作列表"""
        safe = []
        for action in [0, 1, 2, 3]:
            dir_vec = self.action_to_dir(action)
            if self.head_valid(obs, dir_vec):
                safe.append(action)
        return safe

    def manhattan(self, a: tuple[int, int], b: tuple[int, int]) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def safe_fallback(self, obs):
        """選擇任一安全動作；若沒有則隨機動作"""
        safe = self.safe_actions(obs)
        return random.choice(safe) if safe else random.randint(0, 3)

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
        head = tuple(obs['head'])
        food = tuple(obs['food'])

        # 只在安全動作中選擇「使距離食物最近」的動作
        safe = self.safe_actions(obs)
        if not safe:
            return random.randint(0, 3)

        best_action = safe[0]
        best_dist = float('inf')
        for action in safe:
            dx, dy = self.action_to_dir(action)
            nx, ny = head[0] + dx, head[1] + dy
            d = self.manhattan((nx, ny), food)
            if d < best_dist:
                best_dist = d
                best_action = action
        return best_action

class RuleBasedAgent(BaseAgent):
    def __init__(self):
        super().__init__("Safe Rule Agent")

    def select_action(self, obs):
        head = obs['head']
        food = obs['food']

        # 1) 先找所有安全動作
        safe_moves = []
        for action in [0, 1, 2, 3]:
            dir_vec = self.action_to_dir(action)
            if self.head_valid(obs, dir_vec):
                safe_moves.append(action)

        if not safe_moves:
            return random.randint(0, 3)

        # 2) 在安全動作中，選擇離食物最近的 (貪婪)
        best_action = safe_moves[0]
        min_dist = float('inf')
        for action in safe_moves:
            dx, dy = self.action_to_dir(action)
            nx, ny = head[0] + dx, head[1] + dy
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

            for (nx, ny), _action, _dir in self.neighbors(pos):
                if not (0 <= nx < w and 0 <= ny < h):
                    continue
                if (nx, ny) in blocked or (nx, ny) in visited:
                    continue
                visited.add((nx, ny))
                queue.append(((nx, ny), path + [(nx, ny)]))
        return []


    def _direction_from_step(self, head, target):
        hx, hy = head
        tx, ty = target
        dx, dy = tx - hx, ty - hy
        return self.dir_to_action((dx, dy))

    def _safe_fallback(self, head, blocked, w, h):
        pseudo_obs = {
            'head': head,
            'body': list(blocked),
            'grid_size': (w, h),
        }
        return self.safe_fallback(pseudo_obs)

class MovingInCirclesAgent(BaseAgent):
    def __init__(self):
        super().__init__("Hamiltonian Cycle Agent")
        self.cycle = None
        self.next_pos = None

    def select_action(self, obs):
        head = tuple(obs["head"])
        w, h = obs["grid_size"]

        if self.cycle is None:
            assert w % 2 == 0, "Hamiltonian cycle requires even width or height"
            self.cycle = self._build_hamiltonian_cycle(w, h)
            self.next_pos = {
                self.cycle[i]: self.cycle[(i + 1) % len(self.cycle)]
                for i in range(len(self.cycle))
            }

        target = self.next_pos[head]
        return self.direction_from_to(head, target)

    def _build_hamiltonian_cycle(self, w, h):
        """
        真正的哈密頓「環」
        假設 w 是偶數
        """
        path = []

        # 1. 從 (0,0) 往下走最左邊一整欄
        for y in range(h):
            path.append((0, y))

        # 2. 其餘區域做蛇形掃描（避開 x=0）
        for y in reversed(range(h)):
            if (h - y) % 2 == 1:
                xs = range(1, w)
            else:
                xs = reversed(range(1, w))
            for x in xs:
                path.append((x, y))

        return path  # 首尾相鄰，自然成環

    def direction_from_to(self, a, b):
        ax, ay = a
        bx, by = b
        dx, dy = bx - ax, by - ay
        return self.dir_to_action((dx, dy))
