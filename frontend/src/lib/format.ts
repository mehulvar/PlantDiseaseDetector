import type { HealthStatus } from "../api/types";

export const STATUS_LABEL: Record<HealthStatus, string> = {
  healthy: "Healthy",
  at_risk: "At risk",
  unhealthy: "Unhealthy",
};

export function percent(value: number, digits = 0): string {
  return `${(value * 100).toFixed(digits)}%`;
}

export function titleCase(text: string): string {
  return text.replace(/\b\w/g, (c) => c.toUpperCase());
}

/** Format a reading compactly: integers stay integers, others keep up to 2 decimals. */
export function formatReading(value: number): string {
  return Number.isInteger(value) ? String(value) : String(Number(value.toFixed(2)));
}

export function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max);
}
