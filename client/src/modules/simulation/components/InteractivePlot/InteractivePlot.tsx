import Konva from "konva";
import { useCallback, useEffect, useMemo, useState } from "react";
import { Circle, Image as KonvaImage, Layer, Rect, Stage } from "react-konva";
import { useSelector } from "react-redux";
import { RootState, useAppDispatch } from "src/store";
import { setConductorCoords } from "src/store/simulation.reducer";
import useImage from "use-image";
import { ShapeType } from "../../types";
import Arrow from "./shapes/Arrow";
import Ring from "./shapes/Ring";

const InteractivePlot: React.FC<{ containerWidthPx: number }> = ({
  containerWidthPx,
}) => {
  const params = useSelector((state: RootState) => state.simulation.params);
  const result = useSelector((state: RootState) => state.simulation.result);
  const dispatch = useAppDispatch();

  // px-cm utils
  const pxPerCm = useMemo(
    () => containerWidthPx / params.bath.x_boundary,
    [containerWidthPx, params.bath.x_boundary],
  );
  const containerHeightPx = useMemo(
    () => params.bath.y_boundary * pxPerCm,
    [params.bath.y_boundary, pxPerCm],
  );
  const pxVectorToCmVector = useCallback(
    (px: { x: number; y: number }) => ({
      x: px.x / pxPerCm,
      y: (containerHeightPx - px.y) / pxPerCm,
    }),
    [containerHeightPx, pxPerCm],
  );
  const cmVectorToPxVector = useCallback(
    (cm: { x: number; y: number }) => ({
      x: cm.x * pxPerCm,
      y: containerHeightPx - cm.y * pxPerCm,
    }),
    [containerHeightPx, pxPerCm],
  );

  const [pos, setPos] = useState({ x: 100, y: 100 });

  // double binding
  useEffect(() => {
    setPos(
      cmVectorToPxVector({ x: params.conductor.x, y: params.conductor.y }),
    );
  }, [cmVectorToPxVector, params.conductor.x, params.conductor.y]);

  // generic handlers
  const onDragEnd = useCallback(
    (e: Konva.KonvaEventObject<DragEvent>) => {
      if (!e.target) {
        return;
      }
      setPos({ x: e.target.x(), y: e.target.y() });
      dispatch(
        setConductorCoords(
          pxVectorToCmVector({ x: e.target.x(), y: e.target.y() }),
        ),
      );
    },
    [dispatch, pxVectorToCmVector],
  );

  const [bgImage] = useImage(result?.imageUrl ?? "");

  return (
    <>
      <Stage width={containerWidthPx} height={containerHeightPx}>
        <Layer>
          {bgImage && (
            <KonvaImage
              image={bgImage}
              width={containerWidthPx}
              height={containerHeightPx}
            />
          )}
          <Rect
            x={0}
            y={0}
            width={containerWidthPx}
            height={containerHeightPx}
            stroke="black"
            strokeWidth={2}
            listening={false}
          />

          {params.conductor.shape.shape_type === ShapeType.RING && (
            <Ring
              x={pos.x}
              y={pos.y}
              outerRadius={params.conductor.shape.outer_radius}
              innerRadius={params.conductor.shape.inner_radius}
              pxPerCm={pxPerCm}
              containerWidthPx={containerWidthPx}
              containerHeightPx={containerHeightPx}
              onDragEnd={onDragEnd}
            />
          )}

          {params.conductor.shape.shape_type === ShapeType.ARROW && (
            <Arrow
              x={pos.x}
              y={pos.y}
              height={params.conductor.shape.height}
              length={params.conductor.shape.length}
              angle={params.conductor.shape.angle}
              pxPerCm={pxPerCm}
              containerWidthPx={containerWidthPx}
              containerHeightPx={containerHeightPx}
              onDragEnd={onDragEnd}
            />
          )}

          <Circle
            x={pos.x}
            y={pos.y}
            radius={3}
            fill="red"
            stroke="black"
            strokeWidth={1}
          />
        </Layer>
      </Stage>
    </>
  );
};

export default InteractivePlot;
