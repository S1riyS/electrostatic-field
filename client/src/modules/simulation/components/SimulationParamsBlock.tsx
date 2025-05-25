import { Card, Col, Form, Row } from "react-bootstrap";
import { useSelector } from "react-redux";
import FloatInput from "src/modules/common/components/FloatInput";
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
                <FloatInput
                  value={params.bath.x_boundary}
                  setValue={(value) =>
                    handleInputChange("bath", "x_boundary", value)
                  }
                />
              </Form.Group>
            </Col>
            <Col>
              <Form.Group>
                <Form.Label>Y Boundary (cm)</Form.Label>
                <FloatInput
                  value={params.bath.y_boundary}
                  setValue={(value) =>
                    handleInputChange("bath", "y_boundary", value)
                  }
                />
              </Form.Group>
            </Col>
          </Row>

          <h5 className="mt-4">Conductor</h5>
          <Row>
            <Col>
              <Form.Group>
                <Form.Label>X (cm)</Form.Label>
                <FloatInput
                  value={params.conductor.x}
                  setValue={(value) =>
                    handleInputChange("conductor", "x", value)
                  }
                />
              </Form.Group>
            </Col>
            <Col>
              <Form.Group>
                <Form.Label>Y (cm)</Form.Label>
                <FloatInput
                  value={params.conductor.y}
                  setValue={(value) =>
                    handleInputChange("conductor", "y", value)
                  }
                />
              </Form.Group>
            </Col>
            <Col>
              <Form.Group>
                <Form.Label>Potential</Form.Label>
                <FloatInput
                  value={params.conductor.potential}
                  setValue={(value) =>
                    handleInputChange("conductor", "potential", value)
                  }
                />
              </Form.Group>
            </Col>
          </Row>

          <h5 className="mt-4">Electrodes</h5>
          <h6 className="text-muted">Potential</h6>
          <Row>
            <Col>
              <Form.Group>
                <Form.Label>Left</Form.Label>
                <FloatInput
                  value={params.electrodes.left_potential}
                  setValue={(value) =>
                    handleInputChange("electrodes", "left_potential", value)
                  }
                />
              </Form.Group>
            </Col>
            <Col>
              <Form.Group>
                <Form.Label>Right</Form.Label>
                <FloatInput
                  value={params.electrodes.right_potential}
                  setValue={(value) =>
                    handleInputChange("electrodes", "right_potential", value)
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
                  <FloatInput
                    value={params.conductor.shape.inner_radius}
                    setValue={(value) =>
                      handleShapeChange("inner_radius", value)
                    }
                  />
                </Form.Group>
              </Col>
              <Col>
                <Form.Group>
                  <Form.Label>Outer Radius (cm)</Form.Label>
                  <FloatInput
                    value={params.conductor.shape.outer_radius}
                    setValue={(value) =>
                      handleShapeChange("outer_radius", value)
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
                  <FloatInput
                    value={params.conductor.shape.height}
                    setValue={(value) => handleShapeChange("height", value)}
                  />
                </Form.Group>
              </Col>
              <Col>
                <Form.Group>
                  <Form.Label>Length (cm)</Form.Label>
                  <FloatInput
                    value={params.conductor.shape.length}
                    setValue={(value) => handleShapeChange("length", value)}
                  />
                </Form.Group>
              </Col>
              <Col>
                <Form.Group>
                  <Form.Label>Angle (rad)</Form.Label>
                  <FloatInput
                    value={params.conductor.shape.angle}
                    setValue={(value) => handleShapeChange("angle", value)}
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
