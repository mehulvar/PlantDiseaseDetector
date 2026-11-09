import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ApiError, assessPlant, fetchMeta, fetchSampleImage, fetchSamples } from "./api/client";
import type { Assessment, Crop, Sample, SoilReadings } from "./api/types";
import AnalyzePanel from "./components/AnalyzePanel";
import FieldExplorer from "./components/FieldExplorer";
import Footer from "./components/Footer";
import Hero from "./components/Hero";
import HowItWorks from "./components/HowItWorks";
import type { CropChoice } from "./components/LeafUpload";
import Results from "./components/Results";
import Toolbar from "./components/Toolbar";
import { useResource } from "./hooks/useResource";
import { ImageInputError, prepareImage } from "./lib/image";
import { countFilled, healthyExample, parseSoilInputs, readingsToInputs, type SoilInputs } from "./lib/soil";

interface ImageState {
  blob: Blob;
  url: string;
  name: string;
}

const EMPTY_SOIL: SoilInputs = {};

function errorMessage(err: unknown): string {
  if (err instanceof ApiError || err instanceof ImageInputError) return err.message;
  return "Something went wrong. Please try again.";
}

export default function App() {
  const meta = useResource(fetchMeta);
  const samples = useResource(fetchSamples);
  const features = useMemo(() => meta.data?.soil_features ?? [], [meta.data]);

  const [image, setImage] = useState<ImageState | null>(null);
  const [crop, setCrop] = useState<CropChoice>("auto");
  const [soilInputs, setSoilInputs] = useState<SoilInputs>(EMPTY_SOIL);
  const [soilErrors, setSoilErrors] = useState<Readonly<Record<string, string>>>({});
  const [result, setResult] = useState<Assessment | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<number | null>(null);
  const resultsRef = useRef<HTMLDivElement>(null);
  const requestId = useRef(0);

  // Release preview object URLs when the image changes or the app unmounts.
  useEffect(
    () => () => {
      if (image) URL.revokeObjectURL(image.url);
    },
    [image],
  );

  useEffect(() => {
    if (!result) return;
    const node = resultsRef.current?.querySelector<HTMLElement>("#results");
    node?.scrollIntoView({ behavior: "smooth", block: "start" });
    node?.focus({ preventScroll: true });
  }, [result]);

  const setImageBlob = useCallback((blob: Blob, name: string) => {
    setImage({ blob, url: URL.createObjectURL(blob), name });
  }, []);

  const runAssessment = useCallback(
    async (imageBlob: Blob | null, readings: SoilReadings | null, cropChoice: CropChoice) => {
      const id = ++requestId.current;
      setBusy(true);
      setError(null);
      try {
        const response = await assessPlant({
          image: imageBlob,
          soil: readings,
          crop: cropChoice === "auto" ? null : (cropChoice as Crop),
        });
        if (id === requestId.current) setResult(response);
      } catch (err) {
        if (id === requestId.current) setError(errorMessage(err));
      } finally {
        if (id === requestId.current) setBusy(false);
      }
    },
    [],
  );

  const onFile = useCallback(
    async (file: File) => {
      setError(null);
      try {
        setImageBlob(await prepareImage(file), file.name);
      } catch (err) {
        setError(errorMessage(err));
      }
    },
    [setImageBlob],
  );

  const onSubmit = useCallback(() => {
    const parsed = parseSoilInputs(soilInputs, features);
    setSoilErrors(parsed.errors);
    if (Object.keys(parsed.errors).length > 0) {
      setError("Please fix the highlighted soil readings.");
      return;
    }
    const hasReadings = Object.keys(parsed.readings).length > 0;
    void runAssessment(image?.blob ?? null, hasReadings ? parsed.readings : null, crop);
  }, [soilInputs, features, image, crop, runAssessment]);

  /** Put a sample's photo and readings into the form; returns the loaded image blob. */
  const loadSample = useCallback(
    async (sample: Sample): Promise<Blob | null> => {
      const blob = sample.image_url ? await fetchSampleImage(sample.image_url) : null;
      if (blob) setImageBlob(blob, `${sample.title} (sample)`);
      else setImage(null);
      setSoilInputs(readingsToInputs(sample.soil));
      setSoilErrors({});
      setCrop("auto");
      return blob;
    },
    [setImageBlob],
  );

  const onRunSample = useCallback(
    async (sample: Sample) => {
      setError(null);
      try {
        const blob = await loadSample(sample);
        await runAssessment(blob, sample.soil, "auto");
      } catch (err) {
        setError(errorMessage(err));
      }
    },
    [loadSample, runAssessment],
  );

  const onEditSample = useCallback(
    async (sample: Sample) => {
      setError(null);
      try {
        await loadSample(sample);
        document.getElementById("analyze")?.scrollIntoView({ behavior: "smooth" });
      } catch (err) {
        setError(errorMessage(err));
      }
    },
    [loadSample],
  );

  const onSoilChange = useCallback((key: string, value: string) => {
    setSoilInputs((prev) => ({ ...prev, [key]: value }));
    setSoilErrors((prev) => {
      if (!(key in prev)) return prev;
      return Object.fromEntries(Object.entries(prev).filter(([k]) => k !== key));
    });
  }, []);

  return (
    <>
      <Toolbar />
      <main>
        <Hero samples={samples.data ?? []} />
        <FieldExplorer
          samples={samples.data ?? []}
          features={features}
          loading={samples.loading}
          error={samples.error}
          onRetry={samples.retry}
          selected={selected}
          onSelect={setSelected}
          busy={busy}
          runError={error}
          onRun={onRunSample}
          onEdit={onEditSample}
        />
        <AnalyzePanel
          crops={meta.data?.crops ?? ["pepper", "potato", "tomato"]}
          features={features}
          previewUrl={image?.url ?? null}
          fileName={image?.name ?? null}
          crop={crop}
          soilValues={soilInputs}
          soilErrors={soilErrors}
          filledCount={countFilled(soilInputs)}
          busy={busy}
          error={error}
          metaError={meta.error}
          onFile={onFile}
          onClearImage={() => setImage(null)}
          onCrop={setCrop}
          onSoilChange={onSoilChange}
          onFillExample={() => {
            setSoilInputs(healthyExample(features));
            setSoilErrors({});
          }}
          onClearSoil={() => {
            setSoilInputs(EMPTY_SOIL);
            setSoilErrors({});
          }}
          onSubmit={onSubmit}
        />
        <div ref={resultsRef}>{result && <Results result={result} features={features} />}</div>
        <HowItWorks meta={meta.data} />
      </main>
      <Footer />
    </>
  );
}
