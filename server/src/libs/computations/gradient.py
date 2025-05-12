import numpy as np


def gradient_magnitude(f: np.ndarray, dx: float, dy: float) -> np.ndarray:
    # Инициализация массивов для производных
    grad_x = np.zeros_like(f)
    grad_y = np.zeros_like(f)

    # Центральные разности для внутренних точек
    grad_x[1:-1, :] = (f[2:, :] - f[:-2, :]) / (2 * dx)
    grad_y[:, 1:-1] = (f[:, 2:] - f[:, :-2]) / (2 * dy)

    # Граничные точки (односторонние разности)
    grad_x[0, :] = (f[1, :] - f[0, :]) / dx
    grad_x[-1, :] = (f[-1, :] - f[-2, :]) / dx

    grad_y[:, 0] = (f[:, 1] - f[:, 0]) / dy
    grad_y[:, -1] = (f[:, -1] - f[:, -2]) / dy

    return np.sqrt(grad_x**2 + grad_y**2)
