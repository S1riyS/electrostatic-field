from abc import ABC, abstractmethod


class Shape(ABC):
    def __init__(self, x: float, y: float):
        super().__init__()
        self.x = x
        self.y = y

    @abstractmethod
    def check_point(self, x: float, y: float) -> bool:
        "Checks if points A(x, y) lies witin shape"
        ...
