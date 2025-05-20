from typing import Tuple

import numpy as np
from numpy.typing import NDArray


def gradient(f: NDArray[np.float64], dx: float, dy: float) -> NDArray[np.float64]:
    grad_x, grad_y = np.gradient(f, dy, dx)
    return np.dstack([grad_x, grad_y])


def gradient_vectors(
    f: NDArray[np.float64],
    dx: float,
    dy: float,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    grad_y, grad_x = np.gradient(f, dy, dx)
    Ex = grad_x
    Ey = grad_y
    return Ex, Ey


def gradient_magnitude(
    f: NDArray[np.float64],
    dx: float,
    dy: float,
) -> NDArray[np.float64]:
    grad_x, grad_y = np.gradient(f, dy, dx)
    magnitude: NDArray[np.float64] = np.sqrt(grad_x**2 + grad_y**2)
    return magnitude
