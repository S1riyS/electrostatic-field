import { createSlice, PayloadAction } from "@reduxjs/toolkit";

import {
  ArrowShape,
  RingShape,
  ShapeType,
  SimulationParams,
  SimulationResult,
} from "src/modules/simulation/types";

type SimulationState = {
  params: SimulationParams;
  result: SimulationResult | null;
};

const initialState: SimulationState = {
  params: {
    bath: {
      x_boundary: 30,
      y_boundary: 20,
    },
    conductor: {
      x: 15,
      y: 10,
      potential: 7.35,
      // shape: {
      //   shape_type: ShapeType.ARROW,
      //   angle: Math.PI / 4,
      //   height: 4,
      //   length: 8,
      // },
      shape: {
        shape_type: ShapeType.RING,
        inner_radius: 4,
        outer_radius: 8,
      },
    },
    electrodes: {
      left_potential: -7,
      right_potential: 7,
    },
  },
  result: null,
};

const simulationSlice = createSlice({
  name: "approximation",
  initialState,
  reducers: {
    setParams(state, action: PayloadAction<SimulationParams>) {
      state.params = action.payload;
    },
    updateBath(state, action: PayloadAction<SimulationParams["bath"]>) {
      state.params.bath = action.payload;
    },
    updateConductor(
      state,
      action: PayloadAction<SimulationParams["conductor"]>,
    ) {
      state.params.conductor = action.payload;
    },
    setConductorCoords(state, action: PayloadAction<{ x: number; y: number }>) {
      state.params.conductor.x = action.payload.x;
      state.params.conductor.y = action.payload.y;
    },
    updateElectrodes(
      state,
      action: PayloadAction<SimulationParams["electrodes"]>,
    ) {
      state.params.electrodes = action.payload;
    },
    setShapeType(state, action: PayloadAction<ShapeType>) {
      if (action.payload === ShapeType.ARROW) {
        state.params.conductor.shape = {
          shape_type: action.payload,
          angle: 0,
          height: 4,
          length: 8,
        };
      } else if (action.payload === ShapeType.RING) {
        state.params.conductor.shape = {
          shape_type: action.payload,
          inner_radius: 4,
          outer_radius: 8,
        };
      }
    },
    setShape(state, action: PayloadAction<RingShape | ArrowShape>) {
      state.params.conductor.shape = action.payload;
    },
    setResult(state, action: PayloadAction<SimulationResult>) {
      state.result = action.payload;
    },
  },
});

export const {
  setParams,
  setShape,
  updateBath,
  setShapeType,
  updateConductor,
  updateElectrodes,
  setConductorCoords,
  setResult,
} = simulationSlice.actions;
export default simulationSlice.reducer;
