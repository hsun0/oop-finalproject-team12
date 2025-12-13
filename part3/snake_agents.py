import random
import pickle
from pathlib import Path
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
        obstacles = [tuple(p) for p in obs.get("obstacles", [])]

        # 將尾巴暫時視為可通行，因為下一步尾巴會移動 (簡化假設)
        blocked = set(body[:-1]) if len(body) > 1 else set()
        blocked |= set(obstacles)

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

class ReinforcementLearningAgent(BaseAgent):
    def __init__(
        self,
        epsilon_start: float = 0.1,
        epsilon_min: float = 0.01,
        alpha: float = 0.5,
        gamma: float = 0.95,
        model_path: str | None = "./rl_agent.pkl",
        eps_decay_ratio: float = 0.5,
        total_training_steps: int | None = None,
    ):
        super().__init__("Reinforcement Learning Agent")
        # 探索率參數
        self.epsilon_start = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon = epsilon_start
        # 學習率/折扣率
        self.alpha = alpha
        self.gamma = gamma
        # Q 表：鍵為 (state_key, action)
        self.Q: dict[tuple, float] = {}
        # 暫存上一個轉移，用於外部呼叫 update
        self._last_state = None
        self._last_action = None
        # 衰減控制：隨總訓練步數而變動
        self.eps_decay_ratio = eps_decay_ratio  # 衰減步數 = total_steps * ratio
        self.total_training_steps = total_training_steps
        self._t = 0  # 目前已更新的步數（呼叫 update 次數）

        # 嘗試自動載入模型（若指定路徑存在）
        try:
            if model_path and Path(model_path).exists():
                with open(model_path, "rb") as f:
                    payload = pickle.load(f)
                self.Q = payload.get("Q", {})
                # 若模型有存 epsilon 資訊則載入，否則維持初始化值
                self.epsilon = payload.get("epsilon", self.epsilon)
                self.epsilon_start = payload.get("epsilon_start", self.epsilon_start)
                self.epsilon_min = payload.get("epsilon_min", self.epsilon_min)
                self.alpha = payload.get("alpha", self.alpha)
                self.gamma = payload.get("gamma", self.gamma)
                self.eps_decay_ratio = payload.get("eps_decay_ratio", self.eps_decay_ratio)
                self.total_training_steps = payload.get("total_training_steps", self.total_training_steps)
                self._t = payload.get("_t", self._t)
        except Exception:
            pass

    def set_total_training_steps(self, total_steps: int):
        """外部設定預計總訓練步數（例如 episodes * steps_per_episode）。"""
        self.total_training_steps = max(1, int(total_steps))

    def _update_epsilon(self):
        """根據已訓練步數與總步數線性衰減 epsilon。"""
        # 決定衰減步數：總步數 * ratio；若未知總步數，給一個合理預設
        decay_steps = int((self.total_training_steps or 10000) * self.eps_decay_ratio)
        decay_steps = max(1, decay_steps)

        # 線性衰減到 epsilon_min
        progress = min(1.0, self._t / decay_steps)
        self.epsilon = self.epsilon_min + (self.epsilon_start - self.epsilon_min) * (1.0 - progress)

    def _state_key(self, obs) -> tuple:
        head = tuple(obs['head'])
        food = tuple(obs['food'])
        return (head, food)

    def _q(self, state_key, action) -> float:
        return self.Q.get((state_key, action), 0.0)

    def _set_q(self, state_key, action, value: float):
        self.Q[(state_key, action)] = value

    def select_action(self, obs):
        # 取得安全動作；若無安全動作則隨機
        safe = self.safe_actions(obs)
        if not safe:
            action = random.randint(0, 3)
            self._last_state = self._state_key(obs)
            self._last_action = action
            return action

        state_key = self._state_key(obs)

        # epsilon-greedy 探索
        if random.random() < self.epsilon:
            action = random.choice(safe)
            self._last_state = state_key
            self._last_action = action
            return action

        # 選擇 Q 值最高的安全動作
        best_action = safe[0]
        best_value = self._q(state_key, best_action)
        for a in safe[1:]:
            qv = self._q(state_key, a)
            if qv > best_value:
                best_value = qv
                best_action = a

        self._last_state = state_key
        self._last_action = best_action
        return best_action

    def update(self, reward: float, next_obs, done: bool):
        """在環境 step 後呼叫以進行 Q-learning 更新與 epsilon 衰減。"""
        if self._last_state is None or self._last_action is None:
            return

        s = self._last_state
        a = self._last_action
        s_next = self._state_key(next_obs)

        # 計算 max_a' Q(s', a')
        safe_next = self.safe_actions(next_obs)
        max_q_next = 0.0
        if safe_next:
            max_q_next = max(self._q(s_next, ap) for ap in safe_next)

        # Q-learning 更新
        old_q = self._q(s, a)
        target = reward + (0.0 if done else self.gamma * max_q_next)
        new_q = old_q + self.alpha * (target - old_q)
        self._set_q(s, a, new_q)

        # 步數累計與 epsilon 衰減
        self._t += 1
        self._update_epsilon()

    # 便利方法：儲存/載入模型
    def save_model(self, model_path: str = "./rl_agent.pkl"):
        payload = {
            "Q": self.Q,
            "epsilon": self.epsilon,
            "epsilon_start": self.epsilon_start,
            "epsilon_min": self.epsilon_min,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "eps_decay_ratio": self.eps_decay_ratio,
            "total_training_steps": self.total_training_steps,
            "_t": self._t,
        }
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        with open(model_path, "wb") as f:
            pickle.dump(payload, f)

    def load_model(self, model_path: str = "./rl_agent.pkl"):
        if not Path(model_path).exists():
            return False
        with open(model_path, "rb") as f:
            payload = pickle.load(f)
        self.Q = payload.get("Q", {})
        self.epsilon = payload.get("epsilon", self.epsilon)
        self.epsilon_start = payload.get("epsilon_start", self.epsilon_start)
        self.epsilon_min = payload.get("epsilon_min", self.epsilon_min)
        self.alpha = payload.get("alpha", self.alpha)
        self.gamma = payload.get("gamma", self.gamma)
        self.eps_decay_ratio = payload.get("eps_decay_ratio", self.eps_decay_ratio)
        self.total_training_steps = payload.get("total_training_steps", self.total_training_steps)
        self._t = payload.get("_t", self._t)
        return True