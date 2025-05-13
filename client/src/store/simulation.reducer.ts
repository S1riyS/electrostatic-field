import { createSlice, PayloadAction } from "@reduxjs/toolkit";
import { SimulationResponse } from "src/modules/simulation/api/types";

import {
  ArrowShape,
  RingShape,
  ShapeType,
  SimulationParams,
} from "src/modules/simulation/types";

type SimulationState = {
  params: SimulationParams;
  result: SimulationResponse | null;
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
      shape: {
        shape_type: ShapeType.ARROW,
        angle: Math.PI / 4,
        height: 5,
        length: 10,
      },
    },
    electrodes: {
      y_lower: 3,
      y_upper: 17,
      potential: 14,
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
          height: 0,
          length: 0,
        };
      } else if (action.payload === ShapeType.RING) {
        state.params.conductor.shape = {
          shape_type: action.payload,
          inner_radius: 0,
          outer_radius: 0,
        };
      }
    },
    setShape(state, action: PayloadAction<RingShape | ArrowShape>) {
      state.params.conductor.shape = action.payload;
    },
    setResult(state, action: PayloadAction<SimulationResponse>) {
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
  setResult,
} = simulationSlice.actions;
export default simulationSlice.reducer;
