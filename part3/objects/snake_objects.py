"""
Facade module: keep legacy imports working by re-exporting
from the new modular files in this package.
"""

from .base import GridObject
from .snake import Snake
from .food import Food, RedFood, GoldFood
from .obstacles import (
    Obstacle,
    SingleObstacle,
    LShapeObstacle,
    CubeObstacle,
    BeehiveObstacle,
    LoafObstacle,
    BoatObstacle,
    TubObstacle,
    BlinkerObstacle,
    ToadObstacle,
    BeaconObstacle,
    GliderObstacle,
)

__all__ = [
    "GridObject",
    "Snake",
    "Food",
    "RedFood",
    "GoldFood",
    "Obstacle",
    "SingleObstacle",
    "LShapeObstacle",
    "CubeObstacle",
    "BeehiveObstacle",
    "LoafObstacle",
    "BoatObstacle",
    "TubObstacle",
    "BlinkerObstacle",
    "ToadObstacle",
    "BeaconObstacle",
    "GliderObstacle",
]