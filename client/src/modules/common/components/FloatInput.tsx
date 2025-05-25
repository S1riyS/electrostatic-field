import React, { useCallback, useEffect, useState } from "react";
import { Form, FormControlProps } from "react-bootstrap";
import { isStrictFloat } from "../utils/utils";

interface FloatInputProps extends FormControlProps {
  value: number;
  setValue: (value: number) => void;
}

const FloatInput: React.FC<FloatInputProps> = ({
  value,
  setValue,
  ...props
}) => {
  const [stringFloat, setStringFloat] = useState("");

  useEffect(() => {
    setStringFloat(value.toString());
  }, [value]);

  const handleBlur = useCallback(() => {
    if (!isStrictFloat(stringFloat)) {
      setStringFloat(value.toString());
    } else {
      setValue(+stringFloat);
    }
  }, [stringFloat, value, setValue]);

  return (
    <Form.Control
      type="text"
      inputMode="decimal"
      {...props}
      value={stringFloat}
      onChange={(e) => setStringFloat(e.target.value)}
      onBlur={handleBlur}
    />
  );
};

export default FloatInput;
