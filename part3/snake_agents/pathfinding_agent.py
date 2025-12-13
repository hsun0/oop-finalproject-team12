import random
from collections import deque
from .base_agent import BaseAgent

class PathfindingAgent(BaseAgent):
    def __init__(self):
        super().__init__("Pathfinding Agent")

    def select_action(self, obs):
        head = tuple(obs["head"])
        food = tuple(obs["food"])
        body = [tuple(p) for p in obs["body"]]
        w, h = obs["grid_size"]
        obstacles = [tuple(p) for p in obs.get("obstacles", [])]
        blocked = set(body[:-1]) if len(body) > 1 else set()
        blocked |= set(obstacles)
        path = self._bfs(head, food, blocked, w, h)
        if len(path) >= 2:
            return self._direction_from_step(head, path[1])
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
