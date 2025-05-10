package vectors

import (
	"fmt"
	"math"
)

type Vector2D struct {
	X float64
	Y float64
}

// New creates a new Vector2D with the given x and y components
func New(x, y float64) Vector2D {
	return Vector2D{X: x, Y: y}
}

// Add adds another vector to this one and returns the result
func (v Vector2D) Add(other Vector2D) Vector2D {
	return Vector2D{
		X: v.X + other.X,
		Y: v.Y + other.Y,
	}
}

// Subtract subtracts another vector from this one and returns the result
func (v Vector2D) Subtract(other Vector2D) Vector2D {
	return Vector2D{
		X: v.X - other.X,
		Y: v.Y - other.Y,
	}
}

// Multiply multiplies this vector by a scalar and returns the result
func (v Vector2D) Multiply(scalar float64) Vector2D {
	return Vector2D{
		X: v.X * scalar,
		Y: v.Y * scalar,
	}
}

// Divide divides this vector by a scalar and returns the result
func (v Vector2D) Divide(scalar float64) Vector2D {
	return Vector2D{
		X: v.X / scalar,
		Y: v.Y / scalar,
	}
}

// Magnitude returns the length (magnitude) of the vector
func (v Vector2D) Magnitude() float64 {
	return math.Sqrt(v.X*v.X + v.Y*v.Y)
}

// Normalize returns a normalized version of the vector (unit vector)
func (v Vector2D) Normalize() Vector2D {
	mag := v.Magnitude()
	if mag == 0 {
		return Vector2D{}
	}
	return v.Divide(mag)
}

// Dot returns the dot product of this vector with another
func (v Vector2D) Dot(other Vector2D) float64 {
	return v.X*other.X + v.Y*other.Y
}

// Distance returns the distance between this vector and another
func (v Vector2D) Distance(other Vector2D) float64 {
	return v.Subtract(other).Magnitude()
}

// Negate returns the negated vector
func (v Vector2D) Negate() Vector2D {
	return Vector2D{
		X: -v.X,
		Y: -v.Y,
	}
}

// ToPoint converts the vector to a point
func (v Vector2D) ToPoint() (float64, float64) {
	return v.X, v.Y
}

// String returns a string representation of the vector
func (v Vector2D) String() string {
	return fmt.Sprintf("(%.2f, %.2f)", v.X, v.Y)
}
