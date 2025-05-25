import { Shape } from "react-konva";
import { ShapeProps } from "./types";

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

export default Ring;
