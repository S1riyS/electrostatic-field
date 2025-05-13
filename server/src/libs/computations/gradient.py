from typing import Tuple
import numpy as np
import math
from numpy.typing import NDArray


def gradient(f: NDArray[np.float64], dx: float, dy: float) -> NDArray[np.float64]:
    grad_x, grad_y = np.gradient(f, dx, dy)
    return np.dstack([grad_x, grad_y])


def gradient_magnitude(f, dx, dy):
    grad_x, grad_y = np.gradient(f, dx, dy)
    return np.sqrt(grad_x**2 + grad_y**2)
