from abc import ABC, abstractmethod


class GridObject(ABC):
    """Base class for anything that lives on the discrete grid."""

    def __init__(self, grid_width, grid_height):
        self.w = grid_width
        self.h = grid_height

    @abstractmethod
    def reset(self):
        """Subclasses provide their own reset behavior."""
        raise NotImplementedError
