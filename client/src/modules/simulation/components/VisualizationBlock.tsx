import { useEffect, useRef, useState } from "react";
import { Card } from "react-bootstrap";
import InteractivePlot from "./InteractivePlot";

const VisualizationBlock = () => {
  const parentRef = useRef(null);
  const [width, setWidth] = useState(0);

  useEffect(() => {
    const element = parentRef.current as HTMLElement | null;
    if (!element) return;
    const resizeObserver = new ResizeObserver((entries) => {
      for (const entry of entries) {
        setWidth(entry.contentRect.width);
      }
    });
    resizeObserver.observe(element);
    setWidth(element.offsetWidth);
    return () => resizeObserver.disconnect();
  }, []);

  return (
    <Card>
      <Card.Header>
        <Card.Title>Visualization</Card.Title>
      </Card.Header>
      <Card.Body ref={parentRef}>
        {/* {result ? <SimulationPlot result={result} /> : <>no results</>} */}
        <InteractivePlot containerWidthPx={width} />
      </Card.Body>
    </Card>
  );
};

export default VisualizationBlock;
