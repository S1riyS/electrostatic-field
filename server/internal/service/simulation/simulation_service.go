package simulation

import (
	"github.com/S1riyS/electrostatic-field/internal/dto"
)

type ISimulationService interface {
	Run(data dto.SimulationRequest) dto.SimulationResponse
}

type simulationService struct {
}

func NewSimulationService() ISimulationService {
	return &simulationService{}
}

func (ss *simulationService) Run(data dto.SimulationRequest) dto.SimulationResponse {
	return dto.SimulationResponse{}
}
