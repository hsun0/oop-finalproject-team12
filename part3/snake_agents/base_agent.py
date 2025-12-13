from abc import ABC, abstractmethod
import random

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
        obstacles = {tuple(p) for p in obs.get('obstacles', [])}
        if (x, y) in obstacles:
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