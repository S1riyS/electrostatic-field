from abc import ABC, abstractmethod


class Shape(ABC):
    def __init__(self, x: float, y: float):
        super().__init__()
        self.x = x
        self.y = y

    @abstractmethod
    def check_surface(self, x: float, y: float) -> bool:
        "Checks if points A(x, y) lies on surface of shape"
        ...

    @abstractmethod
    def check_inside(self, x: float, y: float) -> bool:
        "Checks if points A(x, y) lies inside outer boundary of shape"
        ...
