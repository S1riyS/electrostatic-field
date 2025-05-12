from libs.shapes.core.shape import Shape


class Arrow(Shape):
    def __init__(self, x: float, y: float, height: float, length: float):
        super().__init__(x, y)
        self.height = height
        self.length = length

    def check_point(self, x: float, y: float) -> bool:
        # Relative to "central point" of shape at (self.x, self.y)
        relative_x = x - self.x
        relative_y = y - self.y

        # Central rect
        central_rect_length = self.length - self.height
        in_central_rect_x = -(central_rect_length / 2) <= relative_x <= (central_rect_length / 2)
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
