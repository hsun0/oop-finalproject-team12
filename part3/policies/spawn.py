class SpawnPolicy:
    def __init__(self, max_cells: int = 40, spawn_chance: float = 0.05, despawn_chance: float = 0.02):
        self.max_cells = max_cells
        self.spawn_chance = spawn_chance
        self.despawn_chance = despawn_chance

    def with_rates(self, *, max_cells=None, spawn_chance=None, despawn_chance=None):
        return SpawnPolicy(
            max_cells=max_cells if max_cells is not None else self.max_cells,
            spawn_chance=spawn_chance if spawn_chance is not None else self.spawn_chance,
            despawn_chance=despawn_chance if despawn_chance is not None else self.despawn_chance,
        )
