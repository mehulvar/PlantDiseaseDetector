import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { Assessment } from "../api/types";
import Results from "./Results";

const result: Assessment = {
  assessment: {
    health_score: 69.9,
    status: "at_risk",
    agreement: "conflict",
    sources: ["image", "soil"],
    contributions: { image: 3.89, soil: -2.6 },
    headline: "Leaf looks healthy, but growing conditions put it at risk",
    explanation: "The two sources disagree.",
  },
  image: {
    prediction: { class_id: "pepper_healthy", name: "Healthy", crop: "pepper", probability: 0.99 },
    top_predictions: [{ class_id: "pepper_healthy", name: "Healthy", crop: "pepper", probability: 0.99 }],
    p_healthy: 0.99,
    predicted_crop: "pepper",
    crop_hint: null,
    low_confidence: false,
    crop_mismatch: false,
    disease: {
      class_id: "pepper_healthy",
      crop: "pepper",
      name: "Healthy",
      healthy: true,
      pathogen: "None",
      pathogen_type: "none",
      summary: "No visible disease.",
      symptoms: ["Uniform green colour"],
      management: [],
    },
  },
  soil: {
    status: "High Stress",
    probabilities: { Healthy: 0, "Moderate Stress": 0, "High Stress": 1 },
    p_healthy: 0,
    provided_features: ["soil_moisture"],
    informative: true,
    drivers: [
      {
        feature: "soil_moisture",
        label: "Soil moisture",
        value: 11,
        unit: "%",
        healthy_low: 30.92,
        healthy_high: 39.26,
        impact: 0.9,
        message: "Soil moisture of 11 % is below the range seen in healthy plants.",
      },
    ],
    out_of_training_range: [],
  },
  recommendations: ["Irrigate deeply."],
  warnings: ["Some warning."],
  disclaimer: "Decision support only.",
};

describe("Results", () => {
  it("shows the verdict, conflict, diagnosis, drivers and advice", () => {
    render(<Results result={result} features={[]} />);
    expect(screen.getByRole("heading", { level: 2 })).toHaveTextContent("At risk");
    expect(screen.getByText(result.assessment.headline)).toBeInTheDocument();
    expect(screen.getByText("Sources disagree")).toBeInTheDocument();
    expect(screen.getByRole("img", { name: "Health score 70 out of 100" })).toBeInTheDocument();
    expect(screen.getByText(/below the range seen in healthy plants/)).toBeInTheDocument();
    expect(screen.getByText("Irrigate deeply.")).toBeInTheDocument();
    expect(screen.getByText("Some warning.")).toBeInTheDocument();
    expect(screen.getByText("Points to healthy")).toBeInTheDocument();
    expect(screen.getByText("Points to a problem")).toBeInTheDocument();
  });

  it("omits the soil card when only a photo was used", () => {
    render(<Results result={{ ...result, soil: null }} features={[]} />);
    expect(screen.queryByText("Soil reading")).not.toBeInTheDocument();
  });
});
