import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import type { SoilFeature } from "../api/types";
import AnalyzePanel from "./AnalyzePanel";

const moisture: SoilFeature = {
  key: "soil_moisture",
  label: "Soil moisture",
  unit: "%",
  group: "soil",
  minimum: 0,
  maximum: 100,
  step: 0.1,
  description: "",
  healthy_low: 30,
  healthy_high: 39,
  typical_min: 10,
  typical_max: 40,
};

function renderPanel(overrides: Partial<Parameters<typeof AnalyzePanel>[0]> = {}) {
  const props = {
    crops: ["pepper", "potato", "tomato"] as const,
    features: [moisture],
    previewUrl: null,
    fileName: null,
    crop: "auto" as const,
    soilValues: {},
    soilErrors: {},
    filledCount: 0,
    busy: false,
    error: null,
    metaError: null,
    onFile: vi.fn(),
    onClearImage: vi.fn(),
    onCrop: vi.fn(),
    onSoilChange: vi.fn(),
    onFillExample: vi.fn(),
    onClearSoil: vi.fn(),
    onSubmit: vi.fn(),
    ...overrides,
  };
  render(<AnalyzePanel {...props} />);
  return props;
}

describe("AnalyzePanel", () => {
  it("disables submit until a photo or reading is provided", () => {
    renderPanel();
    expect(screen.getByRole("button", { name: /assess plant/i })).toBeDisabled();
  });

  it("submits a soil-only analysis", async () => {
    const props = renderPanel({ soilValues: { soil_moisture: "20" }, filledCount: 1 });
    await userEvent.click(screen.getByRole("button", { name: /assess plant/i }));
    expect(props.onSubmit).toHaveBeenCalledOnce();
    expect(screen.getByText(/Single-source analysis/)).toBeInTheDocument();
  });

  it("reports typed soil values and crop choices", async () => {
    const props = renderPanel();
    await userEvent.type(screen.getByLabelText(/Soil moisture/), "5");
    expect(props.onSoilChange).toHaveBeenCalledWith("soil_moisture", "5");
    await userEvent.click(screen.getByRole("radio", { name: /tomato/i }));
    expect(props.onCrop).toHaveBeenCalledWith("tomato");
  });

  it("uploads a chosen photo", async () => {
    const props = renderPanel();
    const file = new File(["leaf"], "leaf.jpg", { type: "image/jpeg" });
    await userEvent.upload(screen.getByLabelText("Upload a leaf photo"), file);
    expect(props.onFile).toHaveBeenCalledWith(file);
  });

  it("shows inline errors and the global error", () => {
    renderPanel({ soilErrors: { soil_moisture: "Must be between 0 and 100." }, error: "Please fix it." });
    expect(screen.getAllByRole("alert").map((n) => n.textContent)).toEqual([
      "Must be between 0 and 100.",
      "Please fix it.",
    ]);
  });
});
