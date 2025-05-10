package shapes

import "github.com/S1riyS/electrostatic-field/pkg/geometry/vectors"

const ARROW_TYPE GeometryType = "arrow"

type Arrow struct {
	Position vectors.Vector2D
	Angle    float64
	H        float64 // Height of an arrow
	L        float64 // Length of an arrow
}
