import numpy as np
from numpy.typing import NDArray

from libs.computations.laplace import DiscretePlanePartition
from schemas.simulation import SimulationElectrode


def get_electrodes_potential_field(
    electrodes_data: SimulationElectrode,
    partition: DiscretePlanePartition,
) -> NDArray[np.float64]:
    # Create meshgrid of all x and y coordinates
    y = np.linspace(0, partition.Ly, partition.Ny)
    x = np.linspace(0, partition.Lx, partition.Nx)
    X, Y = np.meshgrid(x, y)

    # Create the potential field using vectorized operations
    mask = (Y >= electrodes_data.y_lower) & (Y <= electrodes_data.y_upper)
    electrodes_potential = np.where(mask, electrodes_data.potential * (X / partition.Lx), 0)

    return electrodes_potential
