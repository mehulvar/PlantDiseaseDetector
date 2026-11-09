import type { SoilFeature } from "../api/types";
import { formatReading } from "../lib/format";
import type { SoilInputs } from "../lib/soil";

const GROUPS: { key: SoilFeature["group"]; title: string }[] = [
  { key: "soil", title: "Soil test" },
  { key: "environment", title: "Weather & light" },
  { key: "plant", title: "Plant sensors" },
];

interface SoilFormProps {
  features: readonly SoilFeature[];
  values: SoilInputs;
  errors: Readonly<Record<string, string>>;
  onChange: (key: string, value: string) => void;
  onFillExample: () => void;
  onClear: () => void;
}

export default function SoilForm({ features, values, errors, onChange, onFillExample, onClear }: SoilFormProps) {
  return (
    <div className="card">
      <div className="card__head">
        <h3 className="card__title display">Soil &amp; sensors</h3>
        <div className="soil-actions">
          <button type="button" className="ghost-btn" onClick={onFillExample}>
            Fill healthy example
          </button>
          <button type="button" className="ghost-btn" onClick={onClear}>
            Clear
          </button>
        </div>
      </div>
      <p className="field-hint" style={{ marginTop: 0, marginBottom: 18 }}>
        Fill in whatever you have — every field is optional. Soil moisture and nitrogen matter most.
      </p>

      {GROUPS.map((group) => {
        const items = features.filter((f) => f.group === group.key);
        if (items.length === 0) return null;
        return (
          <fieldset key={group.key} className="soil-group" style={{ border: 0, padding: 0, margin: 0 }}>
            <legend className="soil-group__title mono">{group.title}</legend>
            <div className="soil-grid">
              {items.map((feature) => {
                const id = `soil-${feature.key}`;
                const error = errors[feature.key];
                return (
                  <div key={feature.key} className={`soil-input${error ? " soil-input--error" : ""}`}>
                    <label className="soil-input__label" htmlFor={id}>
                      {feature.label}
                      <span className="soil-input__unit mono">{feature.unit}</span>
                    </label>
                    <input
                      id={id}
                      type="number"
                      inputMode="decimal"
                      step="any"
                      min={feature.minimum}
                      max={feature.maximum}
                      placeholder="—"
                      value={values[feature.key] ?? ""}
                      aria-invalid={Boolean(error)}
                      aria-describedby={`${id}-hint`}
                      onChange={(e) => onChange(feature.key, e.target.value)}
                    />
                    {error ? (
                      <span className="soil-input__error" id={`${id}-hint`} role="alert">
                        {error}
                      </span>
                    ) : (
                      <span className="soil-input__hint mono" id={`${id}-hint`} title={feature.description}>
                        Healthy {formatReading(feature.healthy_low)}–{formatReading(feature.healthy_high)}
                      </span>
                    )}
                  </div>
                );
              })}
            </div>
          </fieldset>
        );
      })}
    </div>
  );
}
