import { describe, expect, it } from "vitest";
import type { SoilFeature } from "../api/types";
import { countFilled, healthyExample, parseSoilInputs, readingsToInputs } from "./soil";

const feature = (key: string, minimum: number, maximum: number): SoilFeature => ({
  key,
  label: key,
  unit: "",
  group: "soil",
  minimum,
  maximum,
  step: 0.1,
  description: "",
  healthy_low: minimum + 1,
  healthy_high: minimum + 3,
  typical_min: minimum,
  typical_max: maximum,
});

const FEATURES = [feature("soil_ph", 0, 14), feature("nitrogen", 0, 500)];

describe("parseSoilInputs", () => {
  it("skips empty fields and parses numbers", () => {
    const result = parseSoilInputs({ soil_ph: " 6.5 ", nitrogen: "" }, FEATURES);
    expect(result.readings).toEqual({ soil_ph: 6.5 });
    expect(result.errors).toEqual({});
  });

  it("reports out-of-range and non-numeric values", () => {
    const result = parseSoilInputs({ soil_ph: "15", nitrogen: "abc" }, FEATURES);
    expect(result.readings).toEqual({});
    expect(result.errors.soil_ph).toMatch(/between 0 and 14/);
    expect(result.errors.nitrogen).toBe("Enter a number.");
  });

  it("ignores keys that are not known features", () => {
    expect(parseSoilInputs({ magic: "1" }, FEATURES).readings).toEqual({});
  });
});

describe("helpers", () => {
  it("round-trips readings to text inputs", () => {
    expect(readingsToInputs({ soil_ph: 6.5 })).toEqual({ soil_ph: "6.5" });
    expect(readingsToInputs(null)).toEqual({});
  });

  it("fills the middle of each healthy range", () => {
    expect(healthyExample(FEATURES)).toEqual({ soil_ph: "2", nitrogen: "2" });
  });

  it("counts only non-blank inputs", () => {
    expect(countFilled({ a: "1", b: " ", c: "" })).toBe(1);
  });
});
