from abc import ABC, abstractmethod
from typing import Iterable
from objects.obstacles import Obstacle
from objects.evolution import EvolutionPolicy

class BaseMap(ABC):
    def __init__(self, grid_size: int):
        self.grid_size = grid_size

    @abstractmethod
    def create_obstacles(self, spawn_policy, evolution_policy: EvolutionPolicy | None = None):
        """Return an Obstacle manager configured for this map."""
        raise NotImplementedError

class EmptyMap(BaseMap):
    def create_obstacles(self, spawn_policy, evolution_policy: EvolutionPolicy | None = None):
        # Disable obstacles by setting max_cells=0 and zero chances
        return Obstacle(self.grid_size, self.grid_size, max_obstacles=0, spawn_chance=0.0, despawn_chance=0.0, evolution_policy=evolution_policy)

class ObstacleMap(BaseMap):
    def create_obstacles(self, spawn_policy, evolution_policy: EvolutionPolicy | None = None):
        # Use provided spawn policy to configure obstacle manager
        return Obstacle(
            self.grid_size,
            self.grid_size,
            max_obstacles=getattr(spawn_policy, "max_cells", 40),
            spawn_chance=getattr(spawn_policy, "spawn_chance", 0.05),
            despawn_chance=getattr(spawn_policy, "despawn_chance", 0.02),
            evolution_policy=evolution_policy,
        )
