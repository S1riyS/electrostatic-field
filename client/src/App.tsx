import "bootstrap/dist/css/bootstrap.min.css";
import { Toaster } from "react-hot-toast";
import { Provider } from "react-redux";
import Navbar from "./modules/common/components/Navbar";
import AppRouter from "./modules/common/utils/AppRouter";
import store from "./store";

function App() {
  return (
    <Provider store={store}>
      <AppRouter>
        <Navbar />
        <Toaster position="top-right" />
      </AppRouter>
    </Provider>
  );
}

export default App;
