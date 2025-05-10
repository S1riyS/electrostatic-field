package dto

type SimulationRequest struct {
	// X-axis values for simulation are in [0; X] interval. Units: cm
	X float64 `json:"x" validate:"required,min=1,max:100" example:"40"`
	// Y-axis values for simulation are in [0; Y] interval. Units: cm
	Y float64 `json:"y" validate:"required,min=1,max=100" example:"20"`
	// Geometry type of a conductor
	GeometryType any `json:"geometry_type" validate:"required,geometry_type_enum"`
	// Object that describes geometry of a conductor
	Geometry any
}

type SimulationResponse struct {
	Data any `json:"data"`
}
