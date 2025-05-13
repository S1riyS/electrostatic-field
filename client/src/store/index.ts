import {
  Action,
  combineReducers,
  configureStore,
  ThunkAction,
} from "@reduxjs/toolkit";
import { useDispatch } from "react-redux";
import { simulationApi } from "src/modules/simulation/api/api";
import simulationReducer from "src/store/simulation.reducer";

const rootReducer = combineReducers({
  [simulationApi.reducerPath]: simulationApi.reducer,
  simulation: simulationReducer,
});

const store = configureStore({
  reducer: rootReducer,
  middleware: (getDefaultMiddleware) =>
    getDefaultMiddleware().concat(simulationApi.middleware),
});
export type RootState = ReturnType<typeof rootReducer>;
export type AppDispatch = typeof store.dispatch;
export const useAppDispatch: () => AppDispatch = useDispatch;
export type AppThunk<ReturnType = void> = ThunkAction<
  ReturnType,
  RootState,
  unknown,
  Action<string>
>;

export default store;
