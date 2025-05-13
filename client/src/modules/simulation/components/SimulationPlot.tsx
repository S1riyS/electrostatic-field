import React, { useMemo } from "react";
import Plot from "react-plotly.js";
import { SimulationResponse } from "../api/types";
import { Matrix } from "../utils/matrix";

interface SimulationPlotProps {
  result: SimulationResponse;
}

const SimulationPlot: React.FC<SimulationPlotProps> = ({ result }) => {
  const matrix = useMemo(() => new Matrix(result.data), [result.data]);

  const dx = useMemo(
    () => result.params.bath.x_boundary / (matrix.columnsCount - 1),
    [matrix, result],
  );

  const dy = useMemo(
    () => result.params.bath.y_boundary / (matrix.rowsCount - 1),
    [matrix, result],
  );

  const x = useMemo(
    () => Array.from({ length: matrix.columnsCount }, (_, i) => i * dx),
    [matrix, dx],
  );

  const y = useMemo(
    () => Array.from({ length: matrix.rowsCount }, (_, i) => i * dy),
    [matrix, dy],
  );

  const z = useMemo(() => matrix.matrix, [matrix]);

  return (
    <Plot
      style={{ width: "100%", height: "600px" }}
      useResizeHandler={true}
      config={{ responsive: true }}
      data={[
        {
          z,
          x,
          y,
          type: "heatmap",
          colorscale: "Viridis", // or "Jet", "Hot", etc.
        },
      ]}
      layout={{
        title: { text: "Simulation Heatmap" },
        xaxis: { title: { text: "X Position" } },
        yaxis: { title: { text: "Y Position" } },
        autosize: true,
        height: 600,
      }}
    />
  );
};

export default SimulationPlot;
