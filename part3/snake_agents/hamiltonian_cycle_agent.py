from .base_agent import BaseAgent

class HamiltonianCycleAgent(BaseAgent):
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
        path = []
        for y in range(h):
            path.append((0, y))
        for y in reversed(range(h)):
            if (h - y) % 2 == 1:
                xs = range(1, w)
            else:
                xs = reversed(range(1, w))
            for x in xs:
                path.append((x, y))
        return path

    def direction_from_to(self, a, b):
        ax, ay = a
        bx, by = b
        dx, dy = bx - ax, by - ay
        return self.dir_to_action((dx, dy))
