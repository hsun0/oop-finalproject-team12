import random
import pickle
from pathlib import Path
from .base_agent import BaseAgent

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
        self.epsilon_start = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon = epsilon_start
        self.alpha = alpha
        self.gamma = gamma
        self.Q: dict[tuple, float] = {}
        self._last_state = None
        self._last_action = None
        self.eps_decay_ratio = eps_decay_ratio
        self.total_training_steps = total_training_steps
        self._t = 0
        try:
            if model_path and Path(model_path).exists():
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
        except Exception:
            pass

    def set_total_training_steps(self, total_steps: int):
        self.total_training_steps = max(1, int(total_steps))

    def _update_epsilon(self):
        decay_steps = int((self.total_training_steps or 10000) * self.eps_decay_ratio)
        decay_steps = max(1, decay_steps)
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
        safe = self.safe_actions(obs)
        if not safe:
            action = random.randint(0, 3)
            self._last_state = self._state_key(obs)
            self._last_action = action
            return action
        state_key = self._state_key(obs)
        if random.random() < self.epsilon:
            action = random.choice(safe)
            self._last_state = state_key
            self._last_action = action
            return action
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
        if self._last_state is None or self._last_action is None:
            return
        s = self._last_state
        a = self._last_action
        s_next = self._state_key(next_obs)
        safe_next = self.safe_actions(next_obs)
        max_q_next = 0.0
        if safe_next:
            max_q_next = max(self._q(s_next, ap) for ap in safe_next)
        old_q = self._q(s, a)
        target = reward + (0.0 if done else self.gamma * max_q_next)
        new_q = old_q + self.alpha * (target - old_q)
        self._set_q(s, a, new_q)
        self._t += 1
        self._update_epsilon()

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
