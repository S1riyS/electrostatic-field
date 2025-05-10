package shapes

import "github.com/S1riyS/electrostatic-field/pkg/geometry/vectors"

const RING_TYPE GeometryType = "ring"

type Ring struct {
	Position    vectors.Vector2D
	InnerRadiud float64
	OuterRadiud float64
}

func NewRing(poisition vectors.Vector2D, innerRadiud, outerRadiud float64) *Ring {
	return &Ring{
		Position:    poisition,
		InnerRadiud: innerRadiud,
		OuterRadiud: outerRadiud,
	}
}

func (r *Ring) CheckPoint(point vectors.Vector2D) bool {
	distanceFromCenter := r.Position.Distance(point)
	return distanceFromCenter >= r.InnerRadiud && distanceFromCenter <= r.OuterRadiud
}
