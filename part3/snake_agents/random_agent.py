from .base_agent import BaseAgent
import random

class RandomAgent(BaseAgent):
    def __init__(self):
        super().__init__("Random Agent")

    def select_action(self, obs):
        # 隨機回傳 0~3
        return random.randint(0, 3)