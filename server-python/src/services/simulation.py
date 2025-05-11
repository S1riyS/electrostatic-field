from schemas.simulation import SimulationRequest, SimulationResponse


class SimulationService:
    async def approximate(self, data: SimulationRequest) -> SimulationResponse:
        return SimulationResponse(data=0)
