import math
from typing import List, Literal, Union

from pydantic import BaseModel, Field

from libs.shapes.core.enums import ShapeType


class SimulationBath(BaseModel):
    x_boundary: float = Field(
        gt=0,
        lt=50,
        examples=[40],
        description="X-axis values for simulation are in [0; X] interval. Units: [cm]",
    )
    y_boundary: float = Field(
        gt=0,
        lt=50,
        examples=[20],
        description="Y-axis values for simulation are in [0; Y] interval. Units: [cm]",
    )


class SimulationRingShape(BaseModel):
    shape_type: Literal[ShapeType.RING] = ShapeType.RING
    inner_radius: float = Field(gt=0, lt=10, examples=[5])
    outer_radius: float = Field(gt=0, lt=20, examples=[10])


class SimulationArrowShape(BaseModel):
    shape_type: Literal[ShapeType.ARROW] = ShapeType.ARROW
    height: float = Field(gt=0, lt=10, examples=[6])
    length: float = Field(gt=0, lt=10, examples=[8])
    angle: float = Field(gt=0, lt=2 * math.pi, examples=[math.pi / 2])


class SimulationConductor(BaseModel):
    x: float = Field(gt=1, lt=50, examples=[20], description="Position along x-axis. Units: [cm]")
    y: float = Field(gt=1, lt=50, examples=[10], description="Position along y-axis. Units: [cm]")
    shape: Union[SimulationRingShape, SimulationArrowShape] = Field(discriminator="shape_type")
    potential: float = Field(gt=0, examples=[7.35])


class SimulationElectrode(BaseModel):
    y_lower: float = Field(..., examples=[2], description="Lower Y bound. Units: [cm]")
    y_upper: float = Field(examples=[16], description="Upper Y bound. Units: [cm]")
    potential: float = Field(gt=0, description="Constant electrode potential. Units: [V]")


class SimulationRequest(BaseModel):
    bath: SimulationBath = Field(description="Bath data")
    conductor: SimulationConductor = Field(description="Conductor data")
    electrodes: SimulationElectrode = Field(
        description="Electrodes data. Located at left and right side of simulation area (x=0 and x=x_boundary)"
    )


class SimulationResponse(BaseModel):
    data: List[List[float]]
