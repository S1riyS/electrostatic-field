import os
from typing import Callable, List, Tuple

import mplcursors
import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import NDArray

from libs.computations.gradient import gradient_magnitude, gradient_vectors
from libs.computations.laplace import (
    BoundaryCondition1D,
    BoundaryConditionData,
    BoundaryConditionType,
    BoundaryOrientation,
    DiscretePlanePartition,
    LaplaceSolver,
)
from libs.shapes.core.shape import Shape
from schemas.simulation import (
    SimulationBath,
    SimulationConductor,
    SimulationElectrode,
    SimulationRequest,
    SimulationRingShape,
)
from services.simulation import SimulationService

ConditionsGeneratorFunction = Callable[
    [SimulationRequest], List[Tuple[BoundaryOrientation, BoundaryConditionData]]
]

NX = 1000
NY = 1000
CALCULATED_ELECTRIC_FIELD = 47.62
IMAGES_DIR = "images"


def __get_request() -> SimulationRequest:
    return SimulationRequest(
        bath=SimulationBath(
            x_boundary=30,
            y_boundary=20,
        ),
        conductor=SimulationConductor(
            x=15,
            y=10,
            potential=7.35,
            shape=SimulationRingShape(inner_radius=5, outer_radius=6),
            # shape=SimulationArrowShape(
            #     height=4,
            #     length=6,
            #     angle=0,
            # ),
        ),
        electrodes=SimulationElectrode(
            left_potential=2,
            right_potential=14,
        ),
    )


def __zero_boundary(_: float) -> float:
    return 0


def __generate_electrode_condition(potential: float) -> BoundaryCondition1D:
    def cond(x: float) -> float:
        return potential

    return cond


def __uniform_condition(
    x_max: float,
    potential_low: float,
    potential_high: float,
) -> BoundaryCondition1D:
    def cond(x: float) -> float:
        return potential_low + (x / x_max) * (potential_high - potential_low)

    return cond


def __one_electrode_conditions(
    request: SimulationRequest,
) -> List[Tuple[BoundaryOrientation, BoundaryConditionData]]:
    """
    Boundary conditions for simulation when potential of positive electrode has potential specified in request
    and zero in negative electrode
    """
    return [
        (
            BoundaryOrientation.TOP,
            BoundaryConditionData(BoundaryConditionType.DIRICHLET, __zero_boundary),
        ),
        (
            BoundaryOrientation.BOTTOM,
            BoundaryConditionData(BoundaryConditionType.DIRICHLET, __zero_boundary),
        ),
        (
            BoundaryOrientation.LEFT,
            BoundaryConditionData(BoundaryConditionType.DIRICHLET, __zero_boundary),
        ),
        (
            BoundaryOrientation.RIGHT,
            BoundaryConditionData(
                BoundaryConditionType.DIRICHLET,
                __generate_electrode_condition(
                    request.electrodes.right_potential,
                ),
            ),
        ),
    ]


def __two_electrodes_conditions(
    request: SimulationRequest,
) -> List[Tuple[BoundaryOrientation, BoundaryConditionData]]:
    """
    Boundary conditions for simulation when potential of positive electrode has potential specified in request divied by 2
    and negative electrode has the same potential, but with opposite sign
    """
    # TODO: change DTO so that it takes left and right potentials separately
    return [
        (
            BoundaryOrientation.TOP,
            BoundaryConditionData(
                BoundaryConditionType.DIRICHLET,
                __uniform_condition(
                    request.bath.x_boundary,
                    request.electrodes.left_potential,
                    request.electrodes.right_potential,
                ),
            ),
        ),
        (
            BoundaryOrientation.BOTTOM,
            BoundaryConditionData(
                BoundaryConditionType.DIRICHLET,
                __uniform_condition(
                    request.bath.x_boundary,
                    request.electrodes.left_potential,
                    request.electrodes.right_potential,
                ),
            ),
        ),
        (
            BoundaryOrientation.LEFT,
            BoundaryConditionData(
                BoundaryConditionType.DIRICHLET,
                __generate_electrode_condition(
                    request.electrodes.left_potential,
                ),
            ),
        ),
        (
            BoundaryOrientation.RIGHT,
            BoundaryConditionData(
                BoundaryConditionType.DIRICHLET,
                __generate_electrode_condition(
                    request.electrodes.right_potential,
                ),
            ),
        ),
    ]


def __separate_potentials_conditions(
    request: SimulationRequest,
) -> List[Tuple[BoundaryOrientation, BoundaryConditionData]]:
    """
    Boundary conditions for simulation when potential of conductor and potential of electrodes are calculated separately
    """
    return []


def __apply_electrodes_potential(
    x: float, y: float, shape: Shape, electric_field_value: float
) -> float:
    if shape.check_inside(x, y):
        return 0

    return 0 + (x / 100) * electric_field_value


def __save_electric_lines_plot(
    potential: NDArray[np.float64], width: float, height: float, filename: str
) -> None:
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    Ex, Ey = gradient_vectors(potential, width / NX, height / NY)
    # threshold = 1e-12
    # Ex[Ex < threshold] = 0
    # Ey[Ey < threshold] = 0
    x_grid = np.linspace(0, width, NX)
    y_grid = np.linspace(0, height, NY)

    X, Y = np.meshgrid(x_grid, y_grid)
    plt.figure(figsize=(25, 17.5))

    print(f"Potential shape: {potential.shape}")
    print(f"Grid shape: x: {X.shape}, y: {Y.shape}")
    print(f"Ex shape: {Ex.shape}, Ey shape: {Ey.shape}")

    plt.streamplot(X, Y, -Ex, -Ey, color="red", density=1, linewidth=1)
    plt.contour(X, Y, potential, levels=20, colors="gray")
    plt.savefig(filename, bbox_inches="tight", dpi=300)
    plt.close()


def __save_heatmap(
    data: NDArray[np.float64], partition: DiscretePlanePartition, filename: str
) -> None:
    """
    Save a 2D numpy array as a heatmap image.

    Parameters:
    -----------
    data : 2D numpy.ndarray
        The data array to visualize
    filename : str
        Output filename (with extension, e.g., 'heatmap.png')
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    plt.figure(figsize=(10, 8))

    # Create grid coordinatesc
    x = np.linspace(0, partition.Lx, partition.Nx)
    y = np.linspace(0, partition.Ly, partition.Ny)
    X, Y = np.meshgrid(x, y)

    heatmap = plt.imshow(
        data,
        extent=(0.0, partition.Lx, 0.0, partition.Ly),
        origin="lower",
        aspect="auto",
    )
    plt.colorbar(heatmap)

    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")

    # Add interactive cursor that shows values
    cursor = mplcursors.cursor(heatmap, hover=True)

    # Customize the displayed text
    @cursor.connect("add")
    def on_add(sel):  # type: ignore
        i, j = sel.target.index
        # Make sure indices are within bounds
        i = min(max(i, 0), data.shape[0] - 1)
        j = min(max(j, 0), data.shape[1] - 1)
        sel.annotation.set_text(
            f"x={X[i, j]:.2f}\ny={Y[i, j]:.2f}\nvalue={data[i, j]:.4f}"
        )
        sel.annotation.get_bbox_patch().set(fc="white", alpha=0.8)

    plt.savefig(filename, bbox_inches="tight", dpi=300)
    plt.show()
    plt.close()


def main() -> None:
    # Different approaches to setup boundary conditions
    approaches: List[Tuple[str, ConditionsGeneratorFunction]] = [
        # ("one_electrode", __one_electrode_conditions),
        ("two_electrodes", __two_electrodes_conditions),
        # ("separate_potentials", __separate_potentials_conditions),
    ]

    # Setup core components
    request = __get_request()
    service = SimulationService()

    # Run simulation with different approaches
    for approach_name, conditions_gen_fn in approaches:
        print(f"Running simulation with {approach_name} approach")

        # Retrieve shape
        shape = service._retrieve_shape(request)
        assert shape is not None

        # Setup Laplce equation solver
        partition = service._setup_plane_partition(request, NX, NY)
        solver = LaplaceSolver(partition)

        # Setup internal condition
        internal_condition = service._setup_internal_condition(
            shape, request.conductor.potential
        )
        solver.add_internal_condition(internal_condition)

        conditions = conditions_gen_fn(request)
        for orientation, cond in conditions:
            solver.add_boundary_condition(orientation, cond)

        # Calcilate potential
        u = solver.solve()

        if approach_name == "separate_potentials":
            x = np.linspace(0, partition.Lx, partition.Nx)
            y = np.linspace(0, partition.Ly, partition.Ny)
            X, Y = np.meshgrid(x, y)
            elctrode_potential_fn = np.vectorize(__apply_electrodes_potential)
            elctrode_potential = elctrode_potential_fn(
                X,
                Y,
                shape=shape,
                electric_field_value=CALCULATED_ELECTRIC_FIELD,
            )

            u += elctrode_potential

        __save_heatmap(
            u,
            partition=partition,
            filename=f"{IMAGES_DIR}/{approach_name}/potential.png",
        )

        # Calculate electric field
        electric_field = gradient_magnitude(u, partition.dx, partition.dy)

        __save_electric_lines_plot(
            potential=u,
            width=30,
            height=20,
            filename=f"{IMAGES_DIR}/{approach_name}/potential_lines.png",
        )

        __save_heatmap(
            electric_field,
            partition=partition,
            filename=f"{IMAGES_DIR}/{approach_name}/electric_field.png",
        )


if __name__ == "__main__":
    main()
