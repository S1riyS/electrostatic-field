package router

import (
	"github.com/S1riyS/electrostatic-field/internal/api/controller"
	service "github.com/S1riyS/electrostatic-field/internal/service/simulation"
	"github.com/gofiber/fiber/v2"
)

func NewSimulationRouter(group fiber.Router) {
	ss := service.NewSimulationService()
	sc := controller.SimulationController{SimulationService: ss}
	group.Post("/simulation", sc.Simulate)
}
