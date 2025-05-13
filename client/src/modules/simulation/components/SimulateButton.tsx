import { useCallback } from "react";
import { useSelector } from "react-redux";
import LoadingButton from "src/modules/common/components/LoadingButton";
import { RootState, useAppDispatch } from "src/store";
import { setResult } from "src/store/simulation.reducer";
import { useSimulateMutation } from "../api/api";

const SimulateButton = () => {
  const [fetch, { isLoading }] = useSimulateMutation();
  const params = useSelector((state: RootState) => state.simulation.params);
  const dispatch = useAppDispatch();

  const submit = useCallback(() => {
    fetch(params)
      .unwrap()
      .then((res) => {
        console.log("res", res);
        dispatch(setResult(res));
      });
  }, [fetch, params, dispatch]);

  return (
    <LoadingButton isLoading={isLoading} onClick={submit}>
      Run
    </LoadingButton>
  );
};

export default SimulateButton;
