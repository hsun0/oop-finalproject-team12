from abc import ABC, abstractmethod

class ObstacleShape(ABC):
    """抽象：定義障礙物的形狀、顏色與是否靜態。"""

    color: tuple[int, int, int] = (120, 120, 120)
    static: bool = False  # True 表示不參與演化

    @abstractmethod
    def shape_cells(self, x: int, y: int):
        """回傳以 (x,y) 為參考點的格子座標清單。"""
        raise NotImplementedError
    

class SingleObstacle(ObstacleShape):
    def shape_cells(self, x: int, y: int):
        return [(x, y)]
    color = (160, 160, 160)
    static = True


class LShapeObstacle(ObstacleShape):
    def shape_cells(self, x: int, y: int):
        return [(x, y), (x, y+1), (x+1, y)]
    color = (100, 149, 237)
    static = True


class CubeObstacle(ObstacleShape):
    def shape_cells(self, x: int, y: int):
        return [(x, y), (x+1, y), (x, y+1), (x+1, y+1)]
    color = (200, 200, 60)
    static = True


class BeehiveObstacle(ObstacleShape):
    def shape_cells(self, x: int, y: int):
        return [(x+1, y+0), (x+2, y+0), (x+0, y+1), (x+3, y+1), (x+1, y+2), (x+2, y+2)]
    color = (255, 165, 0)
    static = True


class LoafObstacle(ObstacleShape):
    def shape_cells(self, x: int, y: int):
        return [(x+1, y+0), (x+2, y+0), (x+0, y+1), (x+3, y+1), (x+1, y+2), (x+3, y+2), (x+2, y+3)]
    color = (34, 139, 34)
    static = True


class BoatObstacle(ObstacleShape):
    def shape_cells(self, x: int, y: int):
        return [(x+0, y+0), (x+1, y+0), (x+0, y+1), (x+2, y+1), (x+1, y+2)]
    color = (70, 130, 180)
    static = True


class TubObstacle(ObstacleShape):
    def shape_cells(self, x: int, y: int):
        return [(x+1, y+0), (x+0, y+1), (x+2, y+1), (x+1, y+2)]
    color = (147, 112, 219)
    static = True


class BlinkerObstacle(ObstacleShape):
    def shape_cells(self, x: int, y: int):
        return [(x+0, y+0), (x+1, y+0), (x+2, y+0)]
    color = (255, 105, 180)


class ToadObstacle(ObstacleShape):
    def shape_cells(self, x: int, y: int):
        return [(x+1, y+0), (x+2, y+0), (x+3, y+0), (x+0, y+1), (x+1, y+1), (x+2, y+1)]
    color = (0, 206, 209)


class BeaconObstacle(ObstacleShape):
    def shape_cells(self, x: int, y: int):
        return [(x+0, y+0), (x+1, y+0), (x+0, y+1), (x+1, y+1), (x+2, y+2), (x+3, y+2), (x+2, y+3), (x+3, y+3)]
    color = (255, 140, 0)


class GliderObstacle(ObstacleShape):
    def shape_cells(self, x: int, y: int):
        return [(x+1, y+0), (x+2, y+1), (x+0, y+2), (x+1, y+2), (x+2, y+2)]
    color = (0, 255, 127)
