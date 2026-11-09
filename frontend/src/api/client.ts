import type { Assessment, Crop, Meta, Sample, SoilReadings } from "./types";

const API_BASE = (import.meta.env.VITE_API_URL ?? "").replace(/\/+$/, "");

export class ApiError extends Error {
  readonly status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

export function apiUrl(path: string): string {
  return `${API_BASE}${path}`;
}

async function readError(response: Response): Promise<string> {
  try {
    const body = await response.json();
    if (typeof body?.detail === "string") return body.detail;
    if (Array.isArray(body?.detail)) {
      return body.detail.map((d: { msg?: string }) => d.msg ?? "Invalid input").join("; ");
    }
  } catch {
    // Body was not JSON; fall through to a generic message.
  }
  if (response.status === 429) return "Too many requests - please wait a minute and try again.";
  if (response.status === 413) return "That image is too large. Please use a photo under 8 MB.";
  return `The server returned an error (${response.status}). Please try again.`;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(apiUrl(path), init);
  } catch {
    throw new ApiError("Could not reach the analysis server. Check your connection and try again.", 0);
  }
  if (!response.ok) throw new ApiError(await readError(response), response.status);
  return (await response.json()) as T;
}

export const fetchMeta = () => request<Meta>("/api/meta");
export const fetchSamples = () => request<Sample[]>("/api/samples");

export interface AssessInput {
  image: Blob | null;
  soil: SoilReadings | null;
  crop: Crop | null;
}

export function buildAssessForm({ image, soil, crop }: AssessInput): FormData {
  const form = new FormData();
  if (image) form.append("image", image, image instanceof File ? image.name : "leaf.jpg");
  const readings = Object.fromEntries(
    Object.entries(soil ?? {}).filter(([, v]) => typeof v === "number" && Number.isFinite(v)),
  );
  if (Object.keys(readings).length > 0) form.append("soil", JSON.stringify(readings));
  if (crop) form.append("crop", crop);
  return form;
}

export function assessPlant(input: AssessInput): Promise<Assessment> {
  return request<Assessment>("/api/assess", { method: "POST", body: buildAssessForm(input) });
}

export async function fetchSampleImage(imageUrl: string): Promise<Blob> {
  let response: Response;
  try {
    // Bypass the HTTP cache: an earlier <img> load may have cached a non-CORS copy.
    response = await fetch(apiUrl(imageUrl), { cache: "reload" });
  } catch {
    throw new ApiError("Could not download the sample image.", 0);
  }
  if (!response.ok) throw new ApiError("Could not download the sample image.", response.status);
  return response.blob();
}
