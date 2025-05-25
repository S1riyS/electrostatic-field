import React, { useCallback, useEffect, useState } from "react";
import { Form } from "react-bootstrap";
import { useDebounce } from "use-debounce";

interface SliderProps {
  value: number;
  setValue: (value: number) => void;
}

const Slider: React.FC<SliderProps> = ({ value, setValue }) => {
  const [internalValue, setInternalValue] = useState(value);

  const [debouncedValue] = useDebounce(internalValue, 100);

  // Update Redux (or parent) only when debounced value changes
  useEffect(() => {
    if (debouncedValue !== value) {
      console.log("Updating value:", debouncedValue);
      setValue(debouncedValue);
    }
  }, [debouncedValue, value, setValue]);

  // Sync internal value if external value changes
  useEffect(() => {
    if (value !== internalValue) {
      setInternalValue(value);
    }
    // eslint-disable-next-line
  }, [value]);

  const handleSliderChange = useCallback(
    (event: React.ChangeEvent<HTMLInputElement>) => {
      const newValue = parseFloat(event.target.value);
      setInternalValue(newValue);
    },
    [],
  );

  return (
    <Form.Range
      min={0}
      max={Math.PI * 2}
      step={0.001}
      value={internalValue}
      onChange={handleSliderChange}
    />
  );
};

export default Slider;
