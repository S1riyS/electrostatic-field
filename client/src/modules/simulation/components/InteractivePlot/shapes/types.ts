import Konva from "konva";

export interface ShapeProps extends Konva.ShapeConfig {
  x: number;
  y: number;
  containerWidthPx: number;
  containerHeightPx: number;
  pxPerCm: number;
}
