// Mirrors backend/app/schemas.py.

export type Crop = "pepper" | "potato" | "tomato";
export type HealthStatus = "healthy" | "at_risk" | "unhealthy";
export type SoilReadings = Partial<Record<string, number>>;

export interface ClassProbability {
  class_id: string;
  name: string;
  crop: string;
  probability: number;
}

export interface DiseaseInfo {
  class_id: string;
  crop: string;
  name: string;
  healthy: boolean;
  pathogen: string;
  pathogen_type: string;
  summary: string;
  symptoms: string[];
  management: string[];
}

export interface ImageResult {
  prediction: ClassProbability;
  top_predictions: ClassProbability[];
  p_healthy: number;
  predicted_crop: string;
  crop_hint: Crop | null;
  low_confidence: boolean;
  crop_mismatch: boolean;
  disease: DiseaseInfo;
}

export interface SoilDriver {
  feature: string;
  label: string;
  value: number;
  unit: string;
  healthy_low: number;
  healthy_high: number;
  impact: number;
  message: string;
}

export interface SoilResult {
  status: string;
  probabilities: Record<string, number>;
  p_healthy: number;
  provided_features: string[];
  informative: boolean;
  drivers: SoilDriver[];
  out_of_training_range: string[];
}

export interface Fusion {
  health_score: number;
  status: HealthStatus;
  agreement: "agree" | "conflict" | "single_source";
  sources: ("image" | "soil")[];
  contributions: Record<string, number>;
  headline: string;
  explanation: string;
}

export interface Assessment {
  assessment: Fusion;
  image: ImageResult | null;
  soil: SoilResult | null;
  recommendations: string[];
  warnings: string[];
  disclaimer: string;
}

export interface SoilFeature {
  key: string;
  label: string;
  unit: string;
  group: "soil" | "environment" | "plant";
  minimum: number;
  maximum: number;
  step: number;
  description: string;
  healthy_low: number;
  healthy_high: number;
  typical_min: number;
  typical_max: number;
}

export interface FusionScenario {
  soil_inputs: string;
  rho: number;
  image_only: { balanced_accuracy: number };
  soil_only: { balanced_accuracy: number };
  fused: { balanced_accuracy: number };
}

export interface Meta {
  crops: Crop[];
  classes: DiseaseInfo[];
  soil_features: SoilFeature[];
  soil_classes: string[];
  metrics: {
    image: { architecture: string; test_accuracy: number; macro_f1: number; test_size: number; classes: number };
    soil: {
      evaluation: string;
      rows: number;
      scenarios: Record<string, { accuracy: number; macro_f1: number; features: string[] }>;
    };
    fusion: { conditions?: Record<string, FusionScenario[]> };
  };
}

export interface Sample {
  id: string;
  title: string;
  description: string;
  crop: Crop | null;
  image_url: string | null;
  expected_label: string | null;
  soil: Record<string, number> | null;
}
