from abc import ABC, abstractmethod
from typing import Callable, Iterable, Set, Tuple


Cell = Tuple[int, int]


class EvolutionPolicy(ABC):
    """策略：決定障礙物物件在一步中的演化結果。"""

    @abstractmethod
    def evolve(self, cells: Set[Cell], neighbors_fn: Callable[[Cell], Iterable[Cell]], in_bounds: Callable[[Cell], bool]) -> Set[Cell]:
        """根據既有活細胞集合，回傳下一步的活細胞集合。"""
        raise NotImplementedError


class NoEvolutionPolicy(EvolutionPolicy):
    """不演化（靜態形狀使用）。"""

    def evolve(self, cells: Set[Cell], neighbors_fn: Callable[[Cell], Iterable[Cell]], in_bounds: Callable[[Cell], bool]) -> Set[Cell]:
        return set(cells)


class ConwayEvolutionPolicy(EvolutionPolicy):
    """康威生命遊戲演化規則。"""

    def evolve(self, cells: Set[Cell], neighbors_fn: Callable[[Cell], Iterable[Cell]], in_bounds: Callable[[Cell], bool]) -> Set[Cell]:
        if not cells:
            return set()
        candidates: Set[Cell] = set(cells)
        for c in list(cells):
            candidates.update(neighbors_fn(c))

        next_cells: Set[Cell] = set()
        for c in candidates:
            if not in_bounds(c):
                continue
            live_neighbors = sum((nbr in cells) for nbr in neighbors_fn(c))
            if c in cells:
                if live_neighbors in (2, 3):
                    next_cells.add(c)
            else:
                if live_neighbors == 3:
                    next_cells.add(c)
        return next_cells
