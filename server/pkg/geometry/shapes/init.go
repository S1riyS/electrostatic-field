package shapes

import "github.com/S1riyS/electrostatic-field/pkg/geometry/vectors"

type GeometryType string

type Shape interface {
	// CheckPoint returns true if point lies within shape
	CheckPoint(vectors.Vector2D) bool
}
