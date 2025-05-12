from fastapi import APIRouter
from schemas.simulation import SimulationRequest, SimulationResponse
from services.simulation import SimulationService

simulation_router = APIRouter(prefix="/simulation", tags=["Simulation"])


@simulation_router.post("/", response_model=SimulationResponse)
async def simulate(data: SimulationRequest):
    service = SimulationService()
    return await service.approximate(data)
