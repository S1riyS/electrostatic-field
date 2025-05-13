from libs.shapes.core.shape import Shape


class Rect(Shape):
    def __init__(self, x: float, y: float, a: float, b: float):
        super().__init__(x, y)
        self.a = a
        self.b = b

    def check_surface(self, x: float, y: float) -> bool:
        relative_x = x - self.x
        relative_y = y - self.y

        return -(self.a / 2) <= relative_x <= (self.a / 2) and -(self.b / 2) <= relative_y <= (self.b / 2)

    def check_inside(self, x: float, y: float) -> bool:
        return self.check_surface(x, y)