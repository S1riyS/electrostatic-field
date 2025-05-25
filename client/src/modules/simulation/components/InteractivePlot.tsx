import Konva from "konva";
import { useCallback, useEffect, useMemo, useState } from "react";
import {
  Circle,
  Image as KonvaImage,
  Layer,
  Rect,
  Shape,
  Stage,
} from "react-konva";
import { useSelector } from "react-redux";
import { RootState, useAppDispatch } from "src/store";
import { setConductorCoords } from "src/store/simulation.reducer";
import useImage from "use-image";
import { ShapeType } from "../types";

interface ShapeProps extends Konva.ShapeConfig {
  x: number;
  y: number;
  containerWidthPx: number;
  containerHeightPx: number;
  pxPerCm: number;
}

interface RingProps extends ShapeProps {
  x: number;
  y: number;
  outerRadius: number;
  innerRadius: number;
}
const Ring: React.FC<RingProps> = ({
  outerRadius,
  innerRadius,
  containerWidthPx,
  containerHeightPx,
  pxPerCm,
  ...props
}) => {
  return (
    <Shape
      fill="lightgray"
      stroke="black"
      strokeWidth={2}
      draggable
      sceneFunc={(context, shape) => {
        context.beginPath();
        context.arc(0, 0, outerRadius * pxPerCm, 0, Math.PI * 2, false);
        context.arc(0, 0, innerRadius * pxPerCm, 0, Math.PI * 2, true);
        context.closePath();
        context.fillStrokeShape(shape);
      }}
      dragBoundFunc={(pos) => {
        const r = outerRadius * pxPerCm;
        return {
          x: Math.max(r, Math.min(containerWidthPx - r, pos.x)),
          y: Math.max(r, Math.min(containerHeightPx - r, pos.y)),
        };
      }}
      {...props}
    />
  );
};

interface ArrowProps extends ShapeProps {
  x: number;
  y: number;
  height: number;
  length: number;
  angle: number;
}
const Arrow: React.FC<ArrowProps> = ({
  height: H,
  length: L,
  angle,
  containerWidthPx,
  containerHeightPx,
  pxPerCm,
  ...props
}) => {
  const a = useMemo(() => (L * pxPerCm) / 4, [L, pxPerCm]);
  const b = useMemo(() => (H * pxPerCm) / 2, [H, pxPerCm]);
  const points = useMemo(
    () => [
      { x: a, y: 0 },
      { x: 2 * a, y: b },
      { x: -a, y: b },
      { x: -2 * a, y: 0 },
      { x: -a, y: -b },
      { x: 2 * a, y: -b },
    ],
    [a, b],
  );
  const rotatedPoints = useMemo(() => {
    const angleRad = angle; // angle should be in radians
    const cosA = Math.cos(-angleRad);
    const sinA = Math.sin(-angleRad);

    return points.map(({ x, y }) => ({
      x: x * cosA - y * sinA,
      y: x * sinA + y * cosA,
    }));
  }, [points, angle]);

  const getBoundingBoxOffsets = useCallback(() => {
    let minX = Infinity,
      maxX = -Infinity,
      minY = Infinity,
      maxY = -Infinity;

    rotatedPoints.forEach(({ x, y }) => {
      minX = Math.min(minX, x);
      maxX = Math.max(maxX, x);
      minY = Math.min(minY, y);
      maxY = Math.max(maxY, y);
    });

    return {
      offsetLeft: -minX,
      offsetRight: maxX,
      offsetTop: -minY,
      offsetBottom: maxY,
    };
  }, [rotatedPoints]);

  const bbox = getBoundingBoxOffsets();
  return (
    <>
      <Shape
        fill="lightgray"
        stroke="black"
        strokeWidth={2}
        draggable
        sceneFunc={(context, shape) => {
          if (!rotatedPoints.length) return;
          context.beginPath();
          context.moveTo(rotatedPoints[0].x, rotatedPoints[0].y);
          rotatedPoints.forEach(({ x, y }) => {
            context.lineTo(x, y);
          });
          context.closePath();
          context.fillStrokeShape(shape);
        }}
        {...props}
        dragBoundFunc={(pos) => {
          return {
            x: Math.max(
              bbox.offsetLeft,
              Math.min(containerWidthPx - bbox.offsetRight, pos.x),
            ),
            y: Math.max(
              bbox.offsetTop,
              Math.min(containerHeightPx - bbox.offsetBottom, pos.y),
            ),
          };
        }}
      />
    </>
  );
};

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
