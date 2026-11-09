import { useEffect, type CSSProperties } from "react";
import type { Sample, SoilFeature } from "../api/types";
import { apiUrl } from "../api/client";
import FieldIllustration, { PIN_POSITIONS } from "./FieldIllustration";
import { ChevronLeft, ChevronRight } from "./Icons";
import SampleCard from "./SampleCard";

const FLY_ZOOM = 1.55;
// Centre the selected pin left of middle so the sample card does not cover it.
const FLY_CENTER_X = 40;
const FLY_CENTER_Y = 46;

interface FieldExplorerProps {
  samples: readonly Sample[];
  features: readonly SoilFeature[];
  loading: boolean;
  error: string | null;
  onRetry: () => void;
  selected: number | null;
  onSelect: (index: number | null) => void;
  busy: boolean;
  runError: string | null;
  onRun: (sample: Sample) => void;
  onEdit: (sample: Sample) => void;
}

function planeStyle(selected: number | null): CSSProperties {
  if (selected === null) return {};
  const pos = PIN_POSITIONS[selected % PIN_POSITIONS.length];
  return {
    "--zoom": FLY_ZOOM,
    "--tx": `${(FLY_CENTER_X - pos.x).toFixed(2)}%`,
    "--ty": `${(FLY_CENTER_Y - pos.y).toFixed(2)}%`,
  } as CSSProperties;
}

export default function FieldExplorer(props: FieldExplorerProps) {
  const { samples, features, loading, error, onRetry, selected, onSelect, busy, runError, onRun, onEdit } = props;
  const count = samples.length;
  const current = selected !== null ? samples[selected] : null;

  const step = (delta: number) => {
    if (count === 0) return;
    const base = selected ?? (delta > 0 ? -1 : 0);
    onSelect((base + delta + count) % count);
  };

  useEffect(() => {
    if (selected === null) return;
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") onSelect(null);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [selected, onSelect]);

  return (
    <section className="field" id="field" aria-labelledby="field-title">
      <div className="field__viewport">
        <div className="field__plane" style={planeStyle(selected)}>
          <FieldIllustration />
          {samples.map((sample, index) => {
            const pos = PIN_POSITIONS[index % PIN_POSITIONS.length];
            return (
              <button
                key={sample.id}
                type="button"
                className="pin"
                style={{ left: `${pos.x}%`, top: `${pos.y}%` }}
                aria-pressed={selected === index}
                aria-label={`Open sample: ${sample.title}`}
                onClick={() => onSelect(selected === index ? null : index)}
              >
                <span className="pin__label mono">{sample.title}</span>
                <span className="pin__head" />
                <span className="pin__stem" />
              </button>
            );
          })}
        </div>
      </div>

      <div className="field__intro">
        <span className="mono bracket section-label">01 — Samples</span>
        <h2 className="display" id="field-title">
          Explore
          <br />
          the field
        </h2>
        <p>
          Each pin is a real case: a held-out leaf photo, a soil reading, or both. Open one and run it.
        </p>
        {loading && <p className="mono">Loading samples…</p>}
        {error && (
          <p className="alert" style={{ pointerEvents: "auto" }}>
            {error}{" "}
            <button type="button" className="ghost-btn" onClick={onRetry}>
              Retry
            </button>
          </p>
        )}
      </div>

      {current && (
        <SampleCard
          key={current.id}
          sample={current}
          features={features}
          busy={busy}
          error={runError}
          onRun={onRun}
          onEdit={onEdit}
          onClose={() => onSelect(null)}
        />
      )}

      {count > 0 && (
        <div className="field__nav" role="group" aria-label="Browse samples">
          <button type="button" onClick={() => step(-1)} aria-label="Previous sample">
            <ChevronLeft />
          </button>
          {current?.image_url ? (
            <img className="field__nav-thumb" src={apiUrl(current.image_url)} alt="" crossOrigin="anonymous" />
          ) : (
            <span className="field__nav-thumb" aria-hidden="true" />
          )}
          <span className="field__nav-label mono bracket" aria-live="polite">
            {current ? `${String((selected ?? 0) + 1).padStart(2, "0")} / ${String(count).padStart(2, "0")}` : "Pick a sample"}
          </span>
          <button type="button" onClick={() => step(1)} aria-label="Next sample">
            <ChevronRight />
          </button>
        </div>
      )}

      <span className="field__hint mono bracket">Click the pins to explore</span>
    </section>
  );
}
