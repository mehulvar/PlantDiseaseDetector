import type { SoilFeature, SoilReadings } from "../api/types";

/** Raw text the user typed into each soil field, keyed by feature key. */
export type SoilInputs = Readonly<Record<string, string>>;

export interface ParsedSoil {
  readings: SoilReadings;
  errors: Readonly<Record<string, string>>;
}

/** Validate typed values against each feature's physical bounds; empty fields are skipped. */
export function parseSoilInputs(inputs: SoilInputs, features: readonly SoilFeature[]): ParsedSoil {
  const readings: Record<string, number> = {};
  const errors: Record<string, string> = {};
  for (const feature of features) {
    const raw = (inputs[feature.key] ?? "").trim();
    if (raw === "") continue;
    const value = Number(raw);
    if (!Number.isFinite(value)) {
      errors[feature.key] = "Enter a number.";
    } else if (value < feature.minimum || value > feature.maximum) {
      errors[feature.key] = `Must be between ${feature.minimum} and ${feature.maximum}.`;
    } else {
      readings[feature.key] = value;
    }
  }
  return { readings, errors };
}

export function readingsToInputs(readings: Readonly<Record<string, number>> | null): SoilInputs {
  return Object.fromEntries(Object.entries(readings ?? {}).map(([k, v]) => [k, String(v)]));
}

/** Middle of the healthy range for every feature - a quick way to try the form. */
export function healthyExample(features: readonly SoilFeature[]): SoilInputs {
  return Object.fromEntries(
    features.map((f) => [f.key, String(Number(((f.healthy_low + f.healthy_high) / 2).toFixed(2)))]),
  );
}

export function countFilled(inputs: SoilInputs): number {
  return Object.values(inputs).filter((v) => v.trim() !== "").length;
}
