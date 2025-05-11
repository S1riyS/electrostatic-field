import math
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np


def arrow_condition(x: float, y: float) -> Tuple[float, bool]:
    # Center of arrow is at (x, y) = (0, 0)
    center_x = 0
    center_y = 0

    h = 6
    l = 8
    potential = 7.35

    relative_x = x - center_x
    relative_y = y - center_y

    # Central rect
    central_rect_length = l - h
    in_central_rect_x = -(central_rect_length / 2) <= relative_x <= (central_rect_length / 2)
    in_central_rect_y = -(h / 2) <= relative_y <= (h / 2)
    in_central_rect = in_central_rect_x and in_central_rect_y

    # Left triangle
    in_left_x = -(l / 2) <= relative_x <= -(central_rect_length / 2)
    in_left_upper = relative_y <= (l / 2) + relative_x
    in_left_lower = relative_y >= -(l / 2) - relative_x
    in_left = in_left_x and in_left_upper and in_left_lower

    # Right triangle
    in_right_x = (central_rect_length / 2) <= relative_x <= (l / 2)
    in_right_y = -(h / 2) <= relative_y <= (h / 2)
    in_right_upper = relative_y >= -(central_rect_length) / 2 + relative_x
    in_right_lower = relative_y <= (central_rect_length) / 2 - relative_x
    in_right = in_right_x and in_right_y and (in_right_upper or in_right_lower)

    in_arrow = in_central_rect or in_left or in_right
    if in_arrow:
        return potential, True

    return 0, False


# Create a grid of points
x_min, x_max = -5, 5
y_min, y_max = -5, 5
resolution = 0.1

x = np.arange(x_min, x_max, resolution)
y = np.arange(y_min, y_max, resolution)
X, Y = np.meshgrid(x, y)

# Evaluate the function at each point
Z = np.zeros_like(X)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        val, _ = arrow_condition(X[i, j], Y[i, j])
        Z[i, j] = val

# Create the plot
plt.figure(figsize=(10, 8))
plt.contourf(X, Y, Z, levels=20, cmap="viridis")
plt.colorbar(label="Potential")
plt.title("Arrow Function Visualization")
plt.xlabel("X")
plt.ylabel("Y")

# Add a point at the center
plt.scatter([0], [0], color="red", label="Center (0, 0)")
plt.legend()

plt.grid(True)
plt.axis("equal")  # Одинаковый масштаб по осям
plt.show()
