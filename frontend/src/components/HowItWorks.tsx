import type { FusionScenario, Meta } from "../api/types";
import { percent } from "../lib/format";

const SCENARIO_LABEL: Record<string, string> = {
  soil_test_kit: "Basic soil-test kit",
  all_features: "All sensors",
  moisture_only: "Soil moisture only",
  environment_only: "Air temp, humidity & light only",
};

function FusionTable({ rows }: { rows: readonly FusionScenario[] }) {
  const scenarios = Object.keys(SCENARIO_LABEL).filter((s) => rows.some((r) => r.soil_inputs === s));
  const rhos = [...new Set(rows.map((r) => r.rho))].sort((a, b) => b - a);
  const cell = (scenario: string, rho: number) => rows.find((r) => r.soil_inputs === scenario && r.rho === rho);
  const imageOnly = rows[0]?.image_only.balanced_accuracy ?? 0;
  return (
    <div className="table-wrap">
      <table>
        <caption className="mono">
          Healthy-vs-unhealthy balanced accuracy on field-quality photos (simulated pairing)
        </caption>
        <thead>
          <tr>
            <th scope="col">Photo +</th>
            {rhos.map((rho) => (
              <th key={rho} scope="col">
                Soil/disease link {percent(rho)}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          <tr>
            <th scope="row">Photo only</th>
            {rhos.map((rho) => (
              <td key={rho}>{percent(imageOnly, 1)}</td>
            ))}
          </tr>
          {scenarios.map((scenario) => (
            <tr key={scenario}>
              <th scope="row">{SCENARIO_LABEL[scenario]}</th>
              {rhos.map((rho) => {
                const value = cell(scenario, rho)?.fused.balanced_accuracy;
                return (
                  <td key={rho}>
                    {value === undefined ? "—" : value > imageOnly + 0.005 ? <strong>{percent(value, 1)}</strong> : percent(value, 1)}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function HowItWorks({ meta }: { meta: Meta | null }) {
  const image = meta?.metrics.image;
  const soil = meta?.metrics.soil.scenarios;
  const field = meta?.metrics.fusion.conditions?.field_quality_photos ?? [];
  const kit = field.find((r) => r.soil_inputs === "soil_test_kit" && r.rho === 1);
  return (
    <section className="section how" id="how" aria-labelledby="how-title">
      <div className="section__head">
        <div>
          <span className="mono bracket section-label">04 — How it works</span>
          <h2 className="section__title display" id="how-title">
            Two readers,
            <br />
            one answer
          </h2>
        </div>
        <p className="section__intro lede muted">
          A photo and a soil test see different things. Each model gives its own opinion, then a
          Bayesian fusion step weighs the evidence.
        </p>
      </div>

      <div className="how__grid">
        <article className="how__card">
          <span className="mono bracket muted">Leaf</span>
          <span className="how__metric">{image ? percent(image.test_accuracy, 2) : "—"}</span>
          <h3 className="display">Image model</h3>
          <p className="muted">
            A MobileNetV3 network fine-tuned on {image?.classes ?? 15} pepper, potato and tomato classes.
            Accuracy on {image?.test_size.toLocaleString() ?? "held-out"} held-out lab photos.
          </p>
        </article>
        <article className="how__card">
          <span className="mono bracket muted">Soil</span>
          <span className="how__metric">{soil ? percent(soil.soil_test_kit.accuracy, 1) : "—"}</span>
          <h3 className="display">Soil model</h3>
          <p className="muted">
            Gradient boosting that accepts any subset of readings. Accuracy from a basic soil-test kit,
            tested on plants it never saw.
          </p>
        </article>
        <article className="how__card">
          <span className="mono bracket muted">Fusion</span>
          <span className="how__metric">
            {kit ? `+${Math.round((kit.fused.balanced_accuracy - kit.image_only.balanced_accuracy) * 100)} pts` : "—"}
          </span>
          <h3 className="display">Combined</h3>
          <p className="muted">
            On blurry, poorly lit field photos, adding a soil test lifts healthy-vs-unhealthy accuracy
            {kit ? ` from ${percent(kit.image_only.balanced_accuracy)} to ${percent(kit.fused.balanced_accuracy)}` : ""}.
          </p>
        </article>
      </div>

      {field.length > 0 && <FusionTable rows={field} />}
      <p className="how__note muted">
        The photo and soil datasets come from different sources, so fusion is tested by pairing each
        held-out leaf with a soil reading whose health matches it a set share of the time (the “link”).
        Lab photos score far higher than real field photos, and the soil dataset is small and likely
        synthetic — treat these as indicative, not field-validated.
      </p>
    </section>
  );
}
