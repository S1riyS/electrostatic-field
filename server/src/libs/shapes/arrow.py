import math
from typing import Tuple

from libs.shapes.core.shape import Shape


class Arrow(Shape):
    def __init__(
        self, x: float, y: float, height: float, length: float, angle: float = 0
    ):
        super().__init__(x, y)
        self.height = height
        self.length = length
        self.angle = angle

    def _rotate_point(self, x: float, y: float) -> Tuple[float, float]:
        """Rotates point A(x, y) relative to center of shape on angle self.angle"""
        # Relative coords
        translated_x = x - self.x
        translated_y = y - self.y

        # Rotate
        cos_a = math.cos(-self.angle)
        sin_a = math.sin(-self.angle)
        rotated_x = translated_x * cos_a - translated_y * sin_a
        rotated_y = translated_x * sin_a + translated_y * cos_a

        return rotated_x, rotated_y

    def check_surface(self, x: float, y: float) -> bool:
        relative_x, relative_y = self._rotate_point(x, y)

        # Central rect
        central_rect_length = self.length - self.height
        in_central_rect_x = (
            -(central_rect_length / 2) <= relative_x <= (central_rect_length / 2)
        )
        in_central_rect_y = -(self.height / 2) <= relative_y <= (self.height / 2)
        in_central_rect = in_central_rect_x and in_central_rect_y

        # Left triangle
        in_left_x = -(self.length / 2) <= relative_x <= -(central_rect_length / 2)
        in_left_upper = relative_y <= (self.length / 2) + relative_x
        in_left_lower = relative_y >= -(self.length / 2) - relative_x
        in_left = in_left_x and in_left_upper and in_left_lower

        # Right triangle
        in_right_x = (central_rect_length / 2) <= relative_x <= (self.length / 2)
        in_right_y = -(self.height / 2) <= relative_y <= (self.height / 2)
        in_right_upper = relative_y >= -(central_rect_length) / 2 + relative_x
        in_right_lower = relative_y <= (central_rect_length) / 2 - relative_x
        in_right = in_right_x and in_right_y and (in_right_upper or in_right_lower)

        in_arrow = in_central_rect or in_left or in_right
        if in_arrow:
            return True

        return False

    def check_inside(self, x: float, y: float) -> bool:
        return self.check_surface(x, y)
