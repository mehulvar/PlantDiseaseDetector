import type { Sample, SoilFeature } from "../api/types";
import { apiUrl } from "../api/client";
import { formatReading } from "../lib/format";
import { CloseIcon } from "./Icons";

interface SampleCardProps {
  sample: Sample;
  features: readonly SoilFeature[];
  busy: boolean;
  error: string | null;
  onRun: (sample: Sample) => void;
  onEdit: (sample: Sample) => void;
  onClose: () => void;
}

export default function SampleCard({ sample, features, busy, error, onRun, onEdit, onClose }: SampleCardProps) {
  const labels = new Map(features.map((f) => [f.key, f]));
  const readings = Object.entries(sample.soil ?? {});
  const inputs = [sample.image_url ? "Photo" : null, readings.length ? "Soil data" : null].filter(Boolean);

  return (
    <aside className="sample-card" aria-label={`Sample: ${sample.title}`}>
      <div className="sample-card__media">
        {sample.image_url ? (
          <img src={apiUrl(sample.image_url)} alt={`Leaf photo for sample ${sample.title}`} />
        ) : (
          <div className="sample-card__no-photo mono">No photo — soil readings only</div>
        )}
        <button type="button" className="sample-card__close" onClick={onClose} aria-label="Close sample">
          <CloseIcon />
        </button>
      </div>
      <span className="mono bracket" style={{ display: "block", marginTop: 14, color: "var(--muted)" }}>
        {inputs.join(" + ")}
      </span>
      <h3 className="display">{sample.title}</h3>
      <p>{sample.description}</p>
      <div className="sample-card__actions">
        <button
          type="button"
          className={`pill-btn${busy ? " pill-btn--busy" : ""}`}
          onClick={() => onRun(sample)}
          disabled={busy}
        >
          {busy ? "Analyzing…" : "Run sample"}
          <span className="pill-btn__dot" aria-hidden="true" />
        </button>
        <button type="button" className="ghost-btn" onClick={() => onEdit(sample)} disabled={busy}>
          Edit in form
        </button>
      </div>
      {error && (
        <p className="alert" role="alert">
          {error}
        </p>
      )}
      {readings.length > 0 && (
        <ul className="chips chips--compact" aria-label="Soil readings">
          {readings.map(([key, value]) => {
            const feature = labels.get(key);
            return (
              <li key={key} className="chip">
                {feature?.label ?? key}: {formatReading(value)}
                {feature?.unit ? ` ${feature.unit}` : ""}
              </li>
            );
          })}
        </ul>
      )}
    </aside>
  );
}
