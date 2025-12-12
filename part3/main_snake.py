import time
from dataclasses import dataclass
from typing import Type

from snake_env import SnakeEnv
from snake_agents import RandomAgent, GreedyAgent, RuleBasedAgent, PathfindingAgent, MovingInCirclesAgent


@dataclass
class EpisodeResult:
    score: int
    steps: int


class GameRunner:

    def __init__(self, render = True):
        self.render = render

    def play(self, agent_cls, episodes= 1):
        env = SnakeEnv(render_mode="human" if self.render else None)
        agent = agent_cls()
        print(f"Agent: {agent.name} ")

        results = []
        for ep in range(episodes):
            obs, _ = env.reset()
            terminated = False
            truncated = False
            step = 0

            while not (terminated or truncated):
                action = agent.select_action(obs)
                obs, reward, terminated, truncated, info = env.step(action)
                step += 1

            result = EpisodeResult(score=env.score, steps=step)
            results.append(result)
            print(f"Episode {ep+1} 完成 | 分數: {result.score} | 步數: {result.steps}")
            time.sleep(0.5)

        env.close()
        return results


def run_demo():
    runner = GameRunner(render=False)

    scripts = [
        ("1. Random Agent", RandomAgent, 1),
        ("2. Greedy Agent", GreedyAgent, 1),
        ("3. Rule-Based Agent", RuleBasedAgent, 1),
        ("4. Pathfinding Agent (BFS)", PathfindingAgent, 1),
        ("5. Moving In Circles Agent", MovingInCirclesAgent, 1),
    ]

    for title, agent_cls, episodes in scripts:
        print(f"\n{title}")
        runner.play(agent_cls, episodes=episodes)


if __name__ == "__main__":
    run_demo()