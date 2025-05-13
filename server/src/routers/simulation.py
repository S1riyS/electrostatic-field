from fastapi import APIRouter
from schemas.simulation import SimulationRequest, SimulationResponse
from services.simulation import SimulationService

simulation_router = APIRouter(prefix="/simulation", tags=["Simulation"])


@simulation_router.post("/")
async def simulate(data: SimulationRequest) -> SimulationResponse:
    service = SimulationService()
    return await service.run(data)
