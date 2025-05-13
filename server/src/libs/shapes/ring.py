from libs.shapes.core.shape import Shape
from libs.shapes.utils import get_distance


class Ring(Shape):
    def __init__(self, x: float, y: float, inner_radius: float, outer_radius: float):
        super().__init__(x, y)
        self.inner_radius = inner_radius
        self.outer_radius = outer_radius

        if self.inner_radius >= self.outer_radius:
            raise ValueError("Inner radius must be less than outer radius")

    def check_point(self, x: float, y: float) -> bool:
        distance = get_distance(x, y, self.x, self.y)
        return distance >= self.inner_radius and distance <= self.outer_radius
