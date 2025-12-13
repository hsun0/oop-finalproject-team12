import random
from .base_agent import BaseAgent

class RuleBasedAgent(BaseAgent):
    def __init__(self):
        super().__init__("Safe Rule Agent")

    def select_action(self, obs):
        head = obs['head']
        food = obs['food']
        safe_moves = []
        for action in [0, 1, 2, 3]:
            dir_vec = self.action_to_dir(action)
            if self.head_valid(obs, dir_vec):
                safe_moves.append(action)
        if not safe_moves:
            return random.randint(0, 3)
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
