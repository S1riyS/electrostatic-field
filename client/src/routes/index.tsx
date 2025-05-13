import React from "react";
import NotFoundPage from "src/modules/common/pages/NotFoundPage";
import MainPage from "src/modules/simulation/pages/MainPage";

export type Route = {
  path: string;
  element: React.ReactNode;
};

const routes: Route[] = [
  { element: <MainPage />, path: "/" },
  { element: <NotFoundPage />, path: "/*" },
];

export default routes;
