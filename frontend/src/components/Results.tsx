import type { Assessment, HealthStatus, ImageResult, SoilFeature, SoilResult } from "../api/types";
import { clamp, formatReading, percent, STATUS_LABEL, titleCase } from "../lib/format";

const STATUS_COLOR: Record<HealthStatus, string> = {
  healthy: "var(--healthy)",
  at_risk: "var(--risk)",
  unhealthy: "var(--unhealthy)",
};
const EVIDENCE_RANGE = 5; // log-odds shown from -5 to +5
const RING_RADIUS = 52;
const RING_CIRCUMFERENCE = 2 * Math.PI * RING_RADIUS;

function ScoreRing({ score, status }: { score: number; status: HealthStatus }) {
  const offset = RING_CIRCUMFERENCE * (1 - clamp(score, 0, 100) / 100);
  return (
    <div className="ring" role="img" aria-label={`Health score ${Math.round(score)} out of 100`}>
      <svg viewBox="0 0 120 120">
        <circle cx="60" cy="60" r={RING_RADIUS} fill="none" stroke="var(--bone-deep)" strokeWidth="10" />
        {score >= 0.5 && (
        <circle
          className="ring__arc"
          cx="60"
          cy="60"
          r={RING_RADIUS}
          fill="none"
          stroke={STATUS_COLOR[status]}
          strokeWidth="10"
          strokeLinecap="round"
          strokeDasharray={RING_CIRCUMFERENCE}
          strokeDashoffset={offset}
        />
        )}
      </svg>
      <div className="ring__value">
        <span className="ring__number">{Math.round(score)}</span>
        <span className="mono">Health score</span>
      </div>
    </div>
  );
}

function EvidenceRow({ label, value }: { label: string; value: number }) {
  const width = (Math.min(Math.abs(value), EVIDENCE_RANGE) / EVIDENCE_RANGE) * 50;
  const positive = value >= 0;
  const style = {
    left: positive ? "50%" : `${50 - width}%`,
    width: `${width}%`,
    background: positive ? "var(--healthy)" : "var(--unhealthy)",
  };
  const verdict = Math.abs(value) < 0.05 ? "No influence" : positive ? "Points to healthy" : "Points to a problem";
  return (
    <div className="evidence__row">
      <span className="mono">{label}</span>
      <div className="evidence__track">
        <div className="evidence__bar" style={style} />
      </div>
      <span className="mono" style={{ color: "var(--muted)" }}>
        {verdict}
      </span>
    </div>
  );
}

function ImageCard({ image }: { image: ImageResult }) {
  const disease = image.disease;
  return (
    <div className="card">
      <div className="card__head">
        <h3 className="card__title display">Leaf diagnosis</h3>
        <span className="mono bracket">{image.crop_hint ? `Crop: ${image.crop_hint}` : "Crop detected"}</span>
      </div>
      <ul className="bars" aria-label="Most likely diagnoses">
        {image.top_predictions.map((p) => (
          <li key={p.class_id}>
            <div className="bar__label">
              <span>
                {titleCase(p.crop)} — {p.name}
              </span>
              <span className="mono">{percent(p.probability, 1)}</span>
            </div>
            <div className="bar__track">
              <div className="bar__fill" style={{ width: percent(p.probability, 1) }} />
            </div>
          </li>
        ))}
      </ul>
      <div className="disease">
        <span className="mono bracket" style={{ color: "var(--muted)" }}>
          {disease.healthy ? "No disease detected" : `${disease.pathogen_type} · ${disease.pathogen}`}
        </span>
        <h4 className="display">{disease.name}</h4>
        <p>{disease.summary}</p>
        <p className="mono" style={{ marginTop: 14 }}>
          {disease.healthy ? "What a healthy leaf shows" : "Symptoms to look for"}
        </p>
        <ul>
          {disease.symptoms.map((s) => (
            <li key={s}>{s}</li>
          ))}
        </ul>
      </div>
    </div>
  );
}

function SoilCard({ soil, features }: { soil: SoilResult; features: readonly SoilFeature[] }) {
  const byKey = new Map(features.map((f) => [f.key, f]));
  return (
    <div className="card">
      <div className="card__head">
        <h3 className="card__title display">Soil reading</h3>
        <span className="mono bracket">{soil.provided_features.length} readings</span>
      </div>
      <ul className="bars" aria-label="Stress level probabilities">
        {Object.entries(soil.probabilities).map(([label, p]) => (
          <li key={label}>
            <div className="bar__label">
              <span>{label}</span>
              <span className="mono">{percent(p, 1)}</span>
            </div>
            <div className="bar__track">
              <div className="bar__fill" style={{ width: percent(p, 1) }} />
            </div>
          </li>
        ))}
      </ul>
      {soil.drivers.length > 0 && (
        <div className="disease">
          <p className="mono" style={{ marginTop: 0 }}>
            What is driving it
          </p>
          <ul className="range">
            {soil.drivers.map((d) => {
              const feature = byKey.get(d.feature);
              const lo = Math.min(feature?.typical_min ?? d.healthy_low, d.value);
              const hi = Math.max(feature?.typical_max ?? d.healthy_high, d.value);
              const span = hi - lo || 1;
              const pos = (v: number) => `${((v - lo) / span) * 100}%`;
              return (
                <li key={d.feature}>
                  <div className="range__head">
                    <strong>{d.label}</strong>
                    <span className="mono">
                      {formatReading(d.value)} {d.unit}
                    </span>
                  </div>
                  <div className="range__track" aria-hidden="true">
                    <div
                      className="range__band"
                      style={{
                        left: pos(d.healthy_low),
                        width: `${((d.healthy_high - d.healthy_low) / span) * 100}%`,
                      }}
                    />
                    <div className="range__marker" style={{ left: pos(d.value) }} />
                  </div>
                  <p className="range__text">{d.message}</p>
                </li>
              );
            })}
          </ul>
        </div>
      )}
    </div>
  );
}

interface ResultsProps {
  result: Assessment;
  features: readonly SoilFeature[];
}

export default function Results({ result, features }: ResultsProps) {
  const { assessment, image, soil } = result;
  const sources = assessment.sources.map((s) => (s === "image" ? "leaf photo" : "soil data")).join(" + ");
  return (
    <section className="section results" id="results" aria-labelledby="results-title" tabIndex={-1}>
      <span className="mono bracket section-label">03 — Result · {sources}</span>
      <div className="verdict">
        <ScoreRing score={assessment.health_score} status={assessment.status} />
        <div>
          <h2 id="results-title" className={`verdict__status display status--${assessment.status}`}>
            {STATUS_LABEL[assessment.status]}
          </h2>
          <p className="verdict__headline">{assessment.headline}</p>
          <p className="verdict__explanation">{assessment.explanation}</p>
        </div>
      </div>

      {result.warnings.length > 0 && (
        <div className="warnings" role="note">
          <strong>Please note</strong>
          <ul>
            {result.warnings.map((w) => (
              <li key={w}>{w}</li>
            ))}
          </ul>
        </div>
      )}

      <div className="results__grid">
        {assessment.sources.length > 0 && (
          <div className="card results__wide">
            <div className="card__head">
              <h3 className="card__title display">How each source voted</h3>
              <span className="mono bracket">
                {assessment.agreement === "agree"
                  ? "Sources agree"
                  : assessment.agreement === "conflict"
                    ? "Sources disagree"
                    : "One decisive source"}
              </span>
            </div>
            <div className="evidence">
              {assessment.sources.map((source) => (
                <EvidenceRow
                  key={source}
                  label={source === "image" ? "Leaf" : "Soil"}
                  value={assessment.contributions[source] ?? 0}
                />
              ))}
              <div className="evidence__scale mono" aria-hidden="true">
                <span>Problem</span>
                <span>Neutral</span>
                <span>Healthy</span>
              </div>
            </div>
          </div>
        )}

        {image && <ImageCard image={image} />}
        {soil && <SoilCard soil={soil} features={features} />}

        <div className="card results__wide">
          <div className="card__head">
            <h3 className="card__title display">What to do</h3>
            <span className="mono bracket">{result.recommendations.length} steps</span>
          </div>
          <ol className="recs">
            {result.recommendations.map((r) => (
              <li key={r}>{r}</li>
            ))}
          </ol>
          <p className="disclaimer">{result.disclaimer}</p>
        </div>
      </div>
    </section>
  );
}
