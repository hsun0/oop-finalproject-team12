import random
from .base_agent import BaseAgent

class GreedyAgent(BaseAgent):
    def __init__(self):
        super().__init__("Greedy Agent")

    def select_action(self, obs):
        head = tuple(obs['head'])
        food = tuple(obs['food'])
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
