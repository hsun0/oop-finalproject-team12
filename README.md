# OOP Group Project 說明


- **Part 1**：環境測試。
- **Part 2**：目的提升 FrozenLake-v1的成功率並維持穩定。
- **Part 3**：自行設計貪食蛇環境，遵守 Gym 的 Env interface（reset/step/render），並透過多個 Agent 與class展示多種OOP 概念。

---

## 環境

- Python 3.10+（使用虛擬環境）
- 套件：
	- `gymnasium`
	- `gymnasium[classic_control]`
	- `numpy`
	- `matplotlib`
	- `pygame`（Part 3 繪圖用）
	- `optuna`（Part 2 超參數）

### 建立與啟動虛擬環境

在 `OOP-Group-Project` 下：

```bash
python -m venv .venv
source .venv/bin/activate  # Windows用 .venv\Scripts\activate
```

### 安裝套件

```bash
cd Gymnasium
pip install -e .


cd ..
pip install -r requirements.txt
```



---

## 執行方式

### Part 1


```bash
cd part1

python mountain_car.py --train --episodes 5000

python mountain_car.py --render --episodes 10
```


---

### Part 2：Frozen Lake

主要程式：
- Train 與 test：`part2/frozen_lake.py`
- 使用 Optuna 進行超參數：`part2/hypera.py`

#### 直接進行Train 與 test

```bash
cd ../part2

# 會先訓練 15000 episode，再以 750 episode進行testing
python frozen_lake.py
```

- 執行完畢後：
	- 會輸出 `frozen_lake8x8.pkl`（Q-table）、`frozen_lake8x8.png`（訓練 moving-average 曲線）。
	- 終端機會顯示測試階段的成功率。

#### 使用Optuna超參數

```bash
cd ../part2
python hypera.py
```

- `hypera.py` 使用 Optuna 對以下超參數搜尋：
	- `min_exploration_rate`
	- `epsilon_decay_rate`
	- `discount_factor_g`
	- `start_learning_rate_a`
	- `min_learning_rate_a`
	- `learning_decay_rate`
- 每次 trial 的超參數與對應成功率會寫入 `results.csv`。

---

### Part 3：Snake OOP Project（自訂環境 + 多個 Agent）

主程式：`part3/main_snake.py`

---

### Part 3 結構



```text
part3/
├── main_snake.py          # 程式的entrance：CLI、GameRunner
│
├── tools/                 # 與環境執行相關的工具
│   ├── snake_env.py       # 自訂 SnakeEnv，使其有Gymnasium Env interface（reset/step/render）
│   └── renderer.py        # 使用 pygame 圖像化
│
├── objects/               # 環境中的各種物件（Abstraction + Inheritence + Encapsulation）
│   ├── base.py            # 物件的abstract class
│   ├── snake.py           # Snake class：管理身體、移動、變長與碰撞
│   ├── food.py            # 一般食物與特殊食物class
│   ├── obstacles.py       # 障礙物：生成、更新與碰撞檢查
│   ├── obstacle_shape.py  # 障礙物形狀定義
│   └── evolution.py       # 障礙物演化規則（Strategy / polymorphism）
│
├── snake_agents/          # 各種 Agent 策略
│   ├── base_agent.py              # BaseAgent 的abstract class
│   ├── random_agent.py            # 隨機移動Agent
│   ├── greedy_agent.py            # 單純往食物最近方向移動
│   ├── rule_based_agent.py        # 有安全檢查的規則的 Agent
│   ├── pathfinding_agent.py       # 使用 BFS 方法找到食物
│   ├── hamiltonian_cycle_agent.py # 依漢米爾頓路徑走格子，避免撞到自己
│   └── reinforcement_learning_agent.py # 用 Q-Learning 更新與模型存取的強化學習 Agent
│
└── else/                   # 原有範例程式
	├── oop_project_env.py
	└── warehouse_robot.py
```

#### Demo：展示多種 Agent（著重 OOP 多型）

```bash
cd ../part3

# 預設 demo（不顯示畫面，會輸出分數）
python main_snake.py --mode demo

# Demo 並啟用障礙物
python main_snake.py --mode demo --obstacles

# Demo + 啟用障礙物 + 顯示 pygame 畫面
python main_snake.py --mode demo --obstacles --render
```

在 demo 模式，程式會依序展示：

1. `RandomAgent`
2. `GreedyAgent`
3. `RuleBasedAgent`
4. `PathfindingAgent (BFS)`
5. `HamiltonianCycleAgent`
6. `ReinforcementLearningAgent`（若事先訓練過則會載入 Q-table）


#### 訓練強化學習 Agent

```bash
cd ../part3

# 基本訓練（不顯示畫面）
python main_snake.py --mode train --episodes 10000

# 啟用障礙物並訓練
python main_snake.py --mode train --episodes 10000 --obstacles

# 訓練時顯示動畫
python main_snake.py --mode train --episodes 2000 --obstacles --render
```

訓練完成後會在 `part3` 輸出 `rl_agent.pkl`，包含 Q-table 與相關超參數，之後在 demo 模式選到 RL 時會自動載入此模型。

---
