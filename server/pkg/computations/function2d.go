package computations

import "github.com/S1riyS/electrostatic-field/pkg/geometry/vectors"

type variable string

const (
	XVar variable = "x"
	YVar variable = "y"
)

const delta = 0.0001

type Function2D func(float64, float64) float64

// Derivate returns artial derivative value at given `point` with respect to variable x or y.
//
// Uses "Central Difference Approximation" with error of O(h^2)
func (f Function2D) Derivative(point vectors.Vector2D, variable variable) float64 {
	var numerator float64
	var denominator = 2 * delta
	if variable == XVar {
		numerator = f(point.X+delta, point.Y) - f(point.X-delta, point.Y)
	} else {
		numerator = f(point.X, point.Y+delta) - f(point.X, point.Y-delta)
	}

	return numerator / denominator
}

func (f Function2D) Gradient(point vectors.Vector2D) vectors.Vector2D {
	return vectors.Vector2D{X: f.Derivative(point, XVar), Y: f.Derivative(point, YVar)}
}
