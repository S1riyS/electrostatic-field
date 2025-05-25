from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from schemas.simulation import SimulationRequest
from services.simulation import SimulationService

simulation_router = APIRouter(prefix="/simulation", tags=["Simulation"])


@simulation_router.post("/", response_class=StreamingResponse)
async def simulate(data: SimulationRequest) -> StreamingResponse:
    service = SimulationService()
    image_buf = await service.run(data)

    return StreamingResponse(image_buf, media_type="image/png")
