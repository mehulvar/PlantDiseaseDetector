import { afterEach, describe, expect, it, vi } from "vitest";
import { ApiError, assessPlant, buildAssessForm } from "./client";

afterEach(() => vi.restoreAllMocks());

describe("buildAssessForm", () => {
  it("includes only provided parts and drops non-finite readings", () => {
    const form = buildAssessForm({
      image: new Blob(["x"], { type: "image/jpeg" }),
      soil: { soil_ph: 6.5, nitrogen: Number.NaN },
      crop: "tomato",
    });
    expect(form.get("image")).toBeInstanceOf(Blob);
    expect(JSON.parse(form.get("soil") as string)).toEqual({ soil_ph: 6.5 });
    expect(form.get("crop")).toBe("tomato");
  });

  it("omits soil entirely when there are no readings", () => {
    const form = buildAssessForm({ image: null, soil: {}, crop: null });
    expect(form.has("soil")).toBe(false);
    expect(form.has("image")).toBe(false);
    expect(form.has("crop")).toBe(false);
  });
});

describe("assessPlant", () => {
  it("surfaces the server's detail message", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(JSON.stringify({ detail: "Invalid soil data - soil_ph: too high" }), { status: 422 }),
    );
    await expect(assessPlant({ image: null, soil: { soil_ph: 20 }, crop: null })).rejects.toMatchObject({
      message: "Invalid soil data - soil_ph: too high",
      status: 422,
    });
  });

  it("gives a friendly message when the server is unreachable", async () => {
    vi.spyOn(globalThis, "fetch").mockRejectedValue(new TypeError("Failed to fetch"));
    const error = await assessPlant({ image: null, soil: { soil_ph: 6 }, crop: null }).catch((e) => e);
    expect(error).toBeInstanceOf(ApiError);
    expect(error.message).toMatch(/Could not reach/);
  });

  it("explains rate limiting even without a JSON body", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(new Response("busy", { status: 429 }));
    await expect(assessPlant({ image: null, soil: { soil_ph: 6 }, crop: null })).rejects.toThrow(/Too many requests/);
  });
});
