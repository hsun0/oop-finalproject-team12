import gymnasium as gym
from gymnasium import spaces
import pygame
from abc import ABC, abstractmethod
from dataclasses import dataclass
from snake_objects import Snake, Food, Obstacle

class RewardPolicy(ABC):
    """Strategy interface to compute rewards/termination for each step."""

    @abstractmethod
    def compute(self, *, ate_food: bool, died: bool, steps: int, max_steps: int, score: int):
        raise NotImplementedError


@dataclass
class ClassicRewardPolicy(RewardPolicy):
    reward_food: float = 1.0
    reward_death: float = -1.0
    step_penalty: float = -0.01

    def compute(self, *, ate_food: bool, died: bool, steps: int, max_steps: int, score: int):
        reward = self.step_penalty
        terminated = False
        truncated = False

        if died:
            reward = self.reward_death
            terminated = True
        elif ate_food:
            reward = self.reward_food
        elif steps >= max_steps:
            truncated = True

        return reward, terminated, truncated


class Renderer:
    """Encapsulate pygame drawing to keep env logic clean."""

    def __init__(self, grid_size, cell_size, fps):
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.fps = fps
        self.window = None
        self.clock = None

    def draw(self, snake, food, obstacles=None):
        if self.window is None:
            pygame.init()
            self.window = pygame.display.set_mode((self.grid_size * self.cell_size, self.grid_size * self.cell_size))
            self.clock = pygame.time.Clock()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        self.window.fill((0, 0, 0))

        fx, fy = food.position
        pygame.draw.rect(
            self.window,
            (255, 0, 0),
            (fx * self.cell_size, fy * self.cell_size, self.cell_size, self.cell_size),
        )

        # 畫障礙物
        for ox, oy in obstacles:
            pygame.draw.rect(
                self.window,
                (100, 100, 100),
                (ox * self.cell_size, oy * self.cell_size, self.cell_size, self.cell_size),
            )

        for i, (bx, by) in enumerate(snake.body):
            color = (0, 255, 0) if i == 0 else (0, 200, 0)
            pygame.draw.rect(
                self.window,
                color,
                (bx * self.cell_size, by * self.cell_size, self.cell_size, self.cell_size),
            )

        pygame.display.flip()
        self.clock.tick(self.fps)

    def close(self):
        if self.window:
            pygame.quit()


class SnakeEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 10}

    def __init__(
        self,
        grid_size=20,
        render_mode=None,
        max_steps=None,
        step_penalty=-0.01,
        reward_policy: RewardPolicy | None = None,
    ):
        super().__init__()
        self.grid_size = grid_size # 網格數量 (20x20)
        self.cell_size = 25        # 每個格子的像素大小
        self.render_mode = render_mode
        self.max_steps = max_steps or grid_size * grid_size * 4

        # Reward policy (Strategy pattern)
        self.reward_policy = reward_policy or ClassicRewardPolicy(
            reward_food=1.0,
            reward_death=-1.0,
            step_penalty=step_penalty,
        )
        
        # Action space: 0=Up, 1=Down, 2=Left, 3=Right
        self.action_space = spaces.Discrete(4)
        
        # Observation space: 
        # 為了讓 Agent 好寫，我們回傳 dict (比較直觀)
        # 實際 RL 通常會展平成 Box，但這是 OOP 專案，Dict 比較好 demo
        self.observation_space = spaces.Dict({
            "head": spaces.Box(0, grid_size, shape=(2,), dtype=int),
            "food": spaces.Box(0, grid_size, shape=(2,), dtype=int),
            "body": spaces.Sequence(spaces.Box(0, grid_size, shape=(2,), dtype=int))
        })

        # 初始化物件
        self.snake = Snake(self.grid_size, self.grid_size)
        self.food = Food(self.grid_size, self.grid_size)
        self.obstacles = Obstacle(self.grid_size, self.grid_size)
        self.renderer = Renderer(self.grid_size, self.cell_size, self.metadata["render_fps"]) if render_mode == "human" else None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.snake.reset()
        self.obstacles.reset()
        self.food.respawn(self.snake.body, self.obstacles.positions)
        self.score = 0
        self.steps = 0
        return self._get_obs(), {}

    def step(self, action):
        # 1. 改變方向
        self.snake.change_direction(action)
        
        # 2. 移動蛇
        self.snake.move()

        # 更新障礙物
        self.obstacles.update(self.snake.body, self.food.position)
        
        self.steps += 1
        ate_food = False
        died = not self.snake.alive

        # 障礙物碰撞
        if not died and self.obstacles and self.obstacles.check_collision(self.snake.head):
            died = True

        if not died and self.snake.head == self.food.position:
            ate_food = True
            self.snake.grow()
            self.food.respawn(self.snake.body, self.obstacles.positions)
            self.score += 1

        reward, terminated, truncated = self.reward_policy.compute(
            ate_food=ate_food,
            died=died,
            steps=self.steps,
            max_steps=self.max_steps,
            score=self.score,
        )

        if self.render_mode == "human" and self.renderer:
            self.render()

        return self._get_obs(), reward, terminated, truncated, {"score": self.score, "steps": self.steps}

    def _get_obs(self):
        """回傳給 Agent 的資訊"""
        return {
            "head": self.snake.head,
            "food": self.food.position,
            "body": list(self.snake.body),
            "grid_size": (self.grid_size, self.grid_size), # 方便 Agent 判斷邊界
            "direction": self.snake.direction,
            "obstacles": list(self.obstacles.positions),
        }

    def render(self):
        if self.renderer:
            self.renderer.draw(self.snake, self.food, self.obstacles.positions)

    def close(self):
        if self.renderer:
            self.renderer.close()