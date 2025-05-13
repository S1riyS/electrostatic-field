import { Card } from "react-bootstrap";
import { useSelector } from "react-redux";
import { RootState } from "src/store";
import SimulationPlot from "./SimulationPlot";

const VisualizationBlock = () => {
  const result = useSelector((state: RootState) => state.simulation.result);

  return (
    <Card>
      <Card.Header>
        <Card.Title>Visualization</Card.Title>
      </Card.Header>
      <Card.Body>
        {result ? <SimulationPlot result={result} /> : <>no results</>}
      </Card.Body>
    </Card>
  );
};

export default VisualizationBlock;
