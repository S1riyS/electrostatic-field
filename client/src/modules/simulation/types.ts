export enum ShapeType {
  RING = "RING",
  ARROW = "ARROW",
}

export interface ConductorShape {
  shape_type: ShapeType;
}

export interface RingShape extends ConductorShape {
  shape_type: ShapeType.RING;
  inner_radius: number; // in cm
  outer_radius: number; // in cm
}

export interface ArrowShape extends ConductorShape {
  shape_type: ShapeType.ARROW;
  height: number; // in cm
  length: number; // in cm
  angle: number; // in rad
}

export type SimulationParams = {
  bath: {
    x_boundary: number; // in cm
    y_boundary: number; // in cm
  };
  conductor: {
    x: number; //Position along x-axis. Units: [cm]
    y: number; //Position along y-axis. Units: [cm]
    potential: number; // Constant conductor potential. Units: [V]
    shape: RingShape | ArrowShape;
  };
  electrodes: {
    y_lower: number; // in cm
    y_upper: number; // in cm
    potential: number;
  };
};
