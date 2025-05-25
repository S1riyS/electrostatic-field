import { createApi, fetchBaseQuery } from "@reduxjs/toolkit/query/react";
import { SimulationResult } from "../types";
import { SimulationRequest } from "./types";

export const simulationApi = createApi({
  reducerPath: "simulationApi",
  baseQuery: fetchBaseQuery({
    baseUrl: (import.meta.env.VITE_API_BASE_URL ?? "") + "/api",
    responseHandler: (response) => response.blob(), // fetch Blob instead of JSON
  }),
  endpoints: (build) => ({
    simulate: build.mutation<SimulationResult, SimulationRequest>({
      query: (data) => ({
        url: "/simulation",
        method: "POST",
        body: data,
      }),
      transformResponse: (response: Blob) => ({
        imageUrl: URL.createObjectURL(response),
      }),
    }),
  }),
});

export const { useSimulateMutation } = simulationApi;
