import { useCallback, useMemo } from "react";
import { Shape } from "react-konva";
import { ShapeProps } from "./types";

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

export default Arrow;
