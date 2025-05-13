import { Col, Container, Row } from "react-bootstrap";
import SimulateButton from "../components/SimulateButton";
import SimulationParamsBlock from "../components/SimulationParamsBlock";
import VisualizationBlock from "../components/VisualizationBlock";

const MainPage = () => {
  return (
    <Container>
      <h2>Главная</h2>
      <Row>
        <Col md={6} lg={4}>
          <SimulationParamsBlock />
          <hr />
          <SimulateButton />
        </Col>
        <Col md={6} lg={8}>
          <VisualizationBlock />
        </Col>
      </Row>
    </Container>
  );
};

export default MainPage;
