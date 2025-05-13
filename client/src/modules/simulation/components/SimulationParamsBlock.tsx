import { Card, Col, Form, Row } from "react-bootstrap";
import { useSelector } from "react-redux";
import { ShapeType, SimulationParams } from "src/modules/simulation/types";
import { RootState, useAppDispatch } from "src/store";
import {
  setShape,
  setShapeType,
  updateBath,
  updateConductor,
  updateElectrodes,
} from "src/store/simulation.reducer";

const SimulationParamsBlock = () => {
  const dispatch = useAppDispatch();
  const params = useSelector((state: RootState) => state.simulation.params);

  const handleInputChange = (
    section: keyof SimulationParams,
    field: string,
    value: number,
  ) => {
    const updated = { ...params[section], [field]: value };
    switch (section) {
      case "bath":
        dispatch(updateBath(updated as SimulationParams["bath"]));
        break;
      case "conductor":
        dispatch(updateConductor(updated as SimulationParams["conductor"]));
        break;
      case "electrodes":
        dispatch(updateElectrodes(updated as SimulationParams["electrodes"]));
        break;
    }
  };

  const handleShapeChange = (field: string, value: number) => {
    const updated = { ...params.conductor.shape, [field]: value };
    dispatch(setShape(updated));
  };

  return (
    <Card>
      <Card.Header>Simulation Parameters</Card.Header>
      <Card.Body>
        <Form>
          <h5>Bath</h5>
          <Row>
            <Col>
              <Form.Group>
                <Form.Label>X Boundary (cm)</Form.Label>
                <Form.Control
                  type="number"
                  value={params.bath.x_boundary}
                  onChange={(e) =>
                    handleInputChange(
                      "bath",
                      "x_boundary",
                      Number(e.target.value),
                    )
                  }
                />
              </Form.Group>
            </Col>
            <Col>
              <Form.Group>
                <Form.Label>Y Boundary (cm)</Form.Label>
                <Form.Control
                  type="number"
                  value={params.bath.y_boundary}
                  onChange={(e) =>
                    handleInputChange(
                      "bath",
                      "y_boundary",
                      Number(e.target.value),
                    )
                  }
                />
              </Form.Group>
            </Col>
          </Row>

          <h5 className="mt-4">Conductor</h5>
          <Row>
            <Col>
              <Form.Group>
                <Form.Label>X Position (cm)</Form.Label>
                <Form.Control
                  type="number"
                  value={params.conductor.x}
                  onChange={(e) =>
                    handleInputChange("conductor", "x", Number(e.target.value))
                  }
                />
              </Form.Group>
            </Col>
            <Col>
              <Form.Group>
                <Form.Label>Y Position (cm)</Form.Label>
                <Form.Control
                  type="number"
                  value={params.conductor.y}
                  onChange={(e) =>
                    handleInputChange("conductor", "y", Number(e.target.value))
                  }
                />
              </Form.Group>
            </Col>
            <Col>
              <Form.Group>
                <Form.Label>Potential</Form.Label>
                <Form.Control
                  type="number"
                  value={params.conductor.potential}
                  onChange={(e) =>
                    handleInputChange(
                      "conductor",
                      "potential",
                      Number(e.target.value),
                    )
                  }
                />
              </Form.Group>
            </Col>
          </Row>

          <h5 className="mt-4">Electrodes</h5>
          <Row>
            <Col>
              <Form.Group>
                <Form.Label>Y Lower (cm)</Form.Label>
                <Form.Control
                  type="number"
                  value={params.electrodes.y_lower}
                  onChange={(e) =>
                    handleInputChange(
                      "electrodes",
                      "y_lower",
                      Number(e.target.value),
                    )
                  }
                />
              </Form.Group>
            </Col>
            <Col>
              <Form.Group>
                <Form.Label>Y Upper (cm)</Form.Label>
                <Form.Control
                  type="number"
                  value={params.electrodes.y_upper}
                  onChange={(e) =>
                    handleInputChange(
                      "electrodes",
                      "y_upper",
                      Number(e.target.value),
                    )
                  }
                />
              </Form.Group>
            </Col>
            <Col>
              <Form.Group>
                <Form.Label>Potential</Form.Label>
                <Form.Control
                  type="number"
                  value={params.electrodes.potential}
                  onChange={(e) =>
                    handleInputChange(
                      "electrodes",
                      "potential",
                      Number(e.target.value),
                    )
                  }
                />
              </Form.Group>
            </Col>
          </Row>

          <h5 className="mt-4">Conductor Shape</h5>
          <Form.Group>
            <Form.Label>Shape Type</Form.Label>
            <Form.Select
              value={params.conductor.shape.shape_type}
              onChange={(e) =>
                dispatch(setShapeType(e.target.value as ShapeType))
              }
            >
              <option value={ShapeType.RING}>Ring</option>
              <option value={ShapeType.ARROW}>Arrow</option>
            </Form.Select>
          </Form.Group>

          {params.conductor.shape.shape_type === ShapeType.RING && (
            <Row className="mt-3">
              <Col>
                <Form.Group>
                  <Form.Label>Inner Radius (cm)</Form.Label>
                  <Form.Control
                    type="number"
                    value={params.conductor.shape.inner_radius}
                    onChange={(e) =>
                      handleShapeChange("inner_radius", Number(e.target.value))
                    }
                  />
                </Form.Group>
              </Col>
              <Col>
                <Form.Group>
                  <Form.Label>Outer Radius (cm)</Form.Label>
                  <Form.Control
                    type="number"
                    value={params.conductor.shape.outer_radius}
                    onChange={(e) =>
                      handleShapeChange("outer_radius", Number(e.target.value))
                    }
                  />
                </Form.Group>
              </Col>
            </Row>
          )}

          {params.conductor.shape.shape_type === ShapeType.ARROW && (
            <Row className="mt-3">
              <Col>
                <Form.Group>
                  <Form.Label>Height (cm)</Form.Label>
                  <Form.Control
                    type="number"
                    value={params.conductor.shape.height}
                    onChange={(e) =>
                      handleShapeChange("height", Number(e.target.value))
                    }
                  />
                </Form.Group>
              </Col>
              <Col>
                <Form.Group>
                  <Form.Label>Length (cm)</Form.Label>
                  <Form.Control
                    type="number"
                    value={params.conductor.shape.length}
                    onChange={(e) =>
                      handleShapeChange("length", Number(e.target.value))
                    }
                  />
                </Form.Group>
              </Col>
              <Col>
                <Form.Group>
                  <Form.Label>Angle (rad)</Form.Label>
                  <Form.Control
                    type="number"
                    value={params.conductor.shape.angle}
                    onChange={(e) =>
                      handleShapeChange("angle", Number(e.target.value))
                    }
                  />
                </Form.Group>
              </Col>
            </Row>
          )}
        </Form>
      </Card.Body>
    </Card>
  );
};

export default SimulationParamsBlock;
