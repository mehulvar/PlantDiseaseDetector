import type { Crop, SoilFeature } from "../api/types";
import type { SoilInputs } from "../lib/soil";
import LeafUpload, { type CropChoice } from "./LeafUpload";
import SoilForm from "./SoilForm";

interface AnalyzePanelProps {
  crops: readonly Crop[];
  features: readonly SoilFeature[];
  previewUrl: string | null;
  fileName: string | null;
  crop: CropChoice;
  soilValues: SoilInputs;
  soilErrors: Readonly<Record<string, string>>;
  filledCount: number;
  busy: boolean;
  error: string | null;
  metaError: string | null;
  onFile: (file: File) => void;
  onClearImage: () => void;
  onCrop: (crop: CropChoice) => void;
  onSoilChange: (key: string, value: string) => void;
  onFillExample: () => void;
  onClearSoil: () => void;
  onSubmit: () => void;
}

export default function AnalyzePanel(props: AnalyzePanelProps) {
  const hasImage = props.previewUrl !== null;
  const hasSoil = props.filledCount > 0;
  const canSubmit = (hasImage || hasSoil) && !props.busy;
  return (
    <section className="section" id="analyze" aria-labelledby="analyze-title">
      <div className="section__head">
        <div>
          <span className="mono bracket section-label">02 — Analyze</span>
          <h2 className="section__title display" id="analyze-title">
            Your plant
          </h2>
        </div>
        <p className="section__intro lede">
          Upload a leaf photo, enter soil readings, or both. Using both gives the most reliable result.
        </p>
      </div>

      <form
        onSubmit={(event) => {
          event.preventDefault();
          if (canSubmit) props.onSubmit();
        }}
        noValidate
      >
        <div className="analyze__grid">
          <LeafUpload
            previewUrl={props.previewUrl}
            fileName={props.fileName}
            crop={props.crop}
            crops={props.crops}
            onFile={props.onFile}
            onClear={props.onClearImage}
            onCrop={props.onCrop}
          />
          {props.features.length > 0 ? (
            <SoilForm
              features={props.features}
              values={props.soilValues}
              errors={props.soilErrors}
              onChange={props.onSoilChange}
              onFillExample={props.onFillExample}
              onClear={props.onClearSoil}
            />
          ) : (
            <div className="card">
              <h3 className="card__title display">Soil &amp; sensors</h3>
              <p className="field-hint">{props.metaError ?? "Loading soil fields…"}</p>
            </div>
          )}
        </div>

        <div className="analyze__submit">
          <div className="mode mono" aria-live="polite">
            <span className={`mode__chip${hasImage ? " mode__chip--on" : ""}`}>Photo {hasImage ? "✓" : "—"}</span>
            <span>+</span>
            <span className={`mode__chip${hasSoil ? " mode__chip--on" : ""}`}>
              Soil {hasSoil ? `✓ ${props.filledCount}` : "—"}
            </span>
            <span>→ {hasImage && hasSoil ? "Combined analysis" : hasImage || hasSoil ? "Single-source analysis" : "Add a photo or readings"}</span>
          </div>
          <button type="submit" className={`pill-btn${props.busy ? " pill-btn--busy" : ""}`} disabled={!canSubmit}>
            {props.busy ? "Analyzing…" : "Assess plant"}
            <span className="pill-btn__dot" aria-hidden="true" />
          </button>
        </div>
        {props.error && (
          <p className="alert" role="alert">
            {props.error}
          </p>
        )}
      </form>
    </section>
  );
}
