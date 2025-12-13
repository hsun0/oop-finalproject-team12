import time
import argparse
import pickle
from pathlib import Path
from dataclasses import dataclass
from typing import Type

from snake_env import SnakeEnv
from snake_agents import *


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
                # Optional Q-learning update, if agent supports it
                if hasattr(agent, "update"):
                    try:
                        agent.update(reward, obs, terminated or truncated)
                    except Exception:
                        pass
                step += 1

            result = EpisodeResult(score=env.score, steps=step)
            results.append(result)
            print(f"Episode {ep+1} 完成 | 分數: {result.score} | 步數: {result.steps}")
            time.sleep(0.5)

        env.close()
        return results


def train_rl(model_path: str = "./rl_agent.pkl", episodes: int = 1000, epsilon: float = 0.2, alpha: float = 0.5, gamma: float = 0.95, epsilon_decay: float = 0.995, render: bool = False):
    """
    訓練 ReinforcementLearningAgent，並將學到的 Q-table 與設定儲存到檔案。
    保持 run_demo 不變，僅新增此訓練函式。
    """
    env = SnakeEnv(render_mode="human" if render else None)

    agent = ReinforcementLearningAgent(epsilon=epsilon, alpha=alpha, gamma=gamma, model_path=None)

    for ep in range(episodes):
        obs, _ = env.reset()
        terminated = False
        truncated = False
        step = 0
        while not (terminated or truncated):
            action = agent.select_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            agent.update(reward, obs, terminated or truncated)
            step += 1
        # epsilon 衰減
        agent.epsilon = max(0.01, agent.epsilon * epsilon_decay)
        if (ep + 1) % max(1, episodes // 10) == 0:
            print(f"訓練進度 {ep+1}/{episodes} | epsilon={agent.epsilon:.4f}")

    env.close()

    payload = {
        "Q": agent.Q,
        "epsilon": agent.epsilon,
        "alpha": agent.alpha,
        "gamma": agent.gamma,
    }
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(payload, f)
    print(f"模型已儲存至: {model_path}")


def run_demo():
    runner = GameRunner(render=True)

    scripts = [
        ("1. Random Agent", RandomAgent, 1),
        ("2. Greedy Agent", GreedyAgent, 1),
        ("3. Rule-Based Agent", RuleBasedAgent, 1),
        ("4. Pathfinding Agent (BFS)", PathfindingAgent, 1),
        ("5. Moving In Circles Agent", MovingInCirclesAgent, 1),
        ("6. Reinforcement Learning Agent", ReinforcementLearningAgent, 1),
    ]

    for title, agent_cls, episodes in scripts:
        print(f"\n{title}")
        runner.play(agent_cls, episodes=episodes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Snake Demo/Training")
    parser.add_argument("--mode", choices=["demo","train"], default="demo", help="選擇執行 run_demo 或訓練 RL")
    parser.add_argument("--episodes", type=int, default=1000, help="訓練回合數（train 模式）")
    parser.add_argument("--model-path", type=str, default="./rl_agent.pkl", help="儲存 RL 模型路徑（train 模式）")
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--epsilon-decay", type=float, default=0.995)
    parser.add_argument("--render", action="store_true", help="訓練時顯示動畫")
    args = parser.parse_args()

    if args.mode == "demo":
        run_demo()  # 保持原樣
    else:
        train_rl(
            model_path=args.model_path,
            episodes=args.episodes,
            epsilon=args.epsilon,
            alpha=args.alpha,
            gamma=args.gamma,
            epsilon_decay=args.epsilon_decay,
            render=args.render,
        )

# python3 main_snake.py --mode train --episodes 10000
# python3 main_snake.py --mode demo