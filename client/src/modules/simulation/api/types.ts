import { SimulationParams } from "../types";

export type SimulationRequest = SimulationParams;

export type SimulationResponse = {
  // TODO
  data: number[][];
  params: SimulationParams; // params used for the simulation
};
