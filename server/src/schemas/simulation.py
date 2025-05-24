import math
from typing import List, Literal, Tuple, Union

from pydantic import BaseModel, Field, model_validator
from typing_extensions import Self

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
    inner_radius: float = Field(ge=0, le=10, examples=[5])
    outer_radius: float = Field(gt=0, le=20, examples=[10])

    @model_validator(mode="after")
    def size_validation(self) -> Self:
        if self.inner_radius >= self.outer_radius:
            raise ValueError("Inner raduis cannot be greater than outer radius")
        return self


class SimulationArrowShape(BaseModel):
    shape_type: Literal[ShapeType.ARROW] = ShapeType.ARROW
    height: float = Field(gt=0, le=15, examples=[6])
    length: float = Field(gt=0, le=15, examples=[8])
    angle: float = Field(ge=0, le=2 * math.pi, examples=[math.pi / 2])

    @model_validator(mode="after")
    def size_validation(self) -> Self:
        if self.height > self.length:
            raise ValueError("Height cannot be greater than length")
        return self


class SimulationConductor(BaseModel):
    x: float = Field(gt=1, lt=50, examples=[20], description="Position along x-axis. Units: [cm]")
    y: float = Field(gt=1, lt=50, examples=[10], description="Position along y-axis. Units: [cm]")
    shape: Union[SimulationRingShape, SimulationArrowShape] = Field(discriminator="shape_type")
    potential: float = Field(gt=0, examples=[7.35])


class SimulationElectrode(BaseModel):
    left_potential: float = Field(
        ge=-100,
        le=100,
        description="Constant potential on the left electrode. Units: [V]",
    )
    right_potential: float = Field(
        ge=-100,
        le=100,
        description="Constant potential on the right electrode. Units: [V]",
    )


class SimulationRequest(BaseModel):
    bath: SimulationBath = Field(description="Bath data")
    conductor: SimulationConductor = Field(description="Conductor data")
    electrodes: SimulationElectrode = Field(
        description="Electrodes data. Located at left and right side of simulation area (x=0 and x=x_boundary)"
    )


class SimulationResponse(BaseModel):
    potential: List[List[float]]
    electric_field: List[List[Tuple[float, float]]]
