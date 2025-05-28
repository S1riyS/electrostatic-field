import { useCallback } from "react";
import toast from "react-hot-toast";
import { useSelector } from "react-redux";
import LoadingButton from "src/modules/common/components/LoadingButton";
import { formatApiError } from "src/modules/common/utils/utils";
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
        dispatch(setResult(res));
      })
      .catch((res) => {
        toast.error(formatApiError(res));
      });
  }, [fetch, params, dispatch]);

  return (
    <LoadingButton isLoading={isLoading} onClick={submit}>
      Запуск
    </LoadingButton>
  );
};

export default SimulateButton;
