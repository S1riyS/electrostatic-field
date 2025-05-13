import { createApi, fetchBaseQuery } from "@reduxjs/toolkit/query/react";
import { SimulationRequest, SimulationResponse } from "./types";

export const simulationApi = createApi({
  reducerPath: "simulationApi",
  baseQuery: fetchBaseQuery({
    baseUrl: (import.meta.env.VITE_API_BASE_URL ?? "") + "/api",
  }),
  endpoints: (build) => ({
    simulate: build.mutation<SimulationResponse, SimulationRequest>({
      query: (data) => ({
        url: "/simulation",
        method: "POST",
        body: data,
      }),
      transformResponse: ({ data }: { data: number[][] }, _, params) => {
        return { data, params };
      },
    }),
  }),
});

export const { useSimulateMutation } = simulationApi;
