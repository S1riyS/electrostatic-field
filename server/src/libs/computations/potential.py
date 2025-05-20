import numpy as np
from libs.computations.laplace import DiscretePlanePartition
from libs.shapes.core.shape import Shape
from numpy.typing import NDArray
from schemas.simulation import SimulationElectrode


def get_electrodes_potential_field(
    electrodes_data: SimulationElectrode,
    partition: DiscretePlanePartition,
    shape: Shape,
) -> NDArray[np.float64]:
    y = np.linspace(0, partition.Ly, partition.Ny)
    x = np.linspace(0, partition.Lx, partition.Nx)
    X, Y = np.meshgrid(x, y)

    y_mask = (Y >= electrodes_data.y_lower) & (Y <= electrodes_data.y_upper)

    if hasattr(shape, "check_inside"):
        if np.any(np.vectorize(shape.check_inside)(X, Y)):
            shape_mask = np.vectorize(shape.check_inside)(X, Y)
            y_mask = y_mask & (~shape_mask)

    electrodes_potential = np.where(
        y_mask, electrodes_data.potential * (X / partition.Lx), 0
    )
    return electrodes_potential
