# Plant Health Detector

Detect plant disease from a **leaf photo**, plant stress from **soil & sensor readings**,
or **both together**. The two sources are fused into one health assessment with
plain-language recommendations, so farmers can use whichever data they have.

```
 leaf photo ──► MobileNetV3 (ONNX) ──► P(healthy) + disease diagnosis ─┐
                                                                       ├─► Bayesian evidence fusion ─► health score,
 soil / sensor readings ──► gradient boosting ──► stress level ────────┘    status, explanation, advice
```

| Part | What it does |
| --- | --- |
| `backend/` | FastAPI service: image + soil models, fusion, disease knowledge base, demo samples |
| `ml/` | Reproducible training & evaluation scripts (PyTorch → ONNX, scikit-learn) |
| `frontend/` | Web app (React + Vite) with demo samples, photo upload and soil-reading form |
| `research/` | The original exploratory scripts, figures and first Keras CNN (kept for reference) |
| `data/` | Soil/sensor dataset (`plant_health_data.csv`) |

## Results

### Leaf-image model
MobileNetV3-Large (ImageNet-pretrained, fine-tuned 8 epochs on Apple-silicon GPU), 15 classes across
pepper, potato and tomato, temperature-calibrated, exported to ONNX (17 MB).

| Model | Classes | Held-out test accuracy |
| --- | --- | --- |
| Original Keras CNN (`research/CNN.py`) | 13 | 88% |
| **MobileNetV3 (this version)** | **15** | **99.85%** (macro-F1 0.999, n = 2,064) |

The test score is measured on the **exported ONNX model using the exact preprocessing the API uses**.
Two classes the original dropped for being small (potato healthy, tomato mosaic virus) are now included
with class-weighted training.

> PlantVillage photos are taken in a lab on plain backgrounds. On *degraded "field-quality"* versions of
> the same test photos (blur, poor exposure, noise, heavy JPEG) healthy/diseased balanced accuracy drops
> to ~65% - which is exactly where soil data helps (see fusion below).

### Soil / sensor model
Histogram gradient boosting trained with **random feature masking**, so it gives sensible answers from
any subset of readings. Evaluated with 5-fold **GroupKFold by plant** (tested on plants it never saw).

| Readings supplied | Accuracy (3 classes) |
| --- | --- |
| All 11 features | 99.6% |
| Basic soil-test kit (moisture, pH, N, P, K) | 99.6% |
| Soil moisture only | 86.0% |
| pH + N/P/K only | 45.5% |
| Air temperature, humidity, light only | 35.3% (≈ chance) |

> The soil dataset (1,200 rows, 10 plants) appears synthetic: stress is driven almost entirely by soil
> moisture and nitrogen, which is why accuracy is near-perfect when those are present. Treat these
> numbers as an upper bound, not field performance.

### Fusion: does combining the two help?
Each model's opinion is converted to evidence (its log-likelihood ratio relative to its own training
prior), weighted (image 1.0, soil 0.5 - images observe disease directly, soil only measures a risk
factor) and summed. Uninformative readings contribute exactly zero. Two safeguards: a single photo can
never count as more than 98% certain, and a plant is never called "healthy" when either source shows a
strong problem (e.g. a healthy-looking leaf in parched soil → "at risk").

**The two datasets are unpaired** (no plant has both a photo and a soil reading), so fusion is evaluated
by *simulated pairing*: every held-out test leaf is paired with an unseen-plant soil reading whose health
state matches the leaf's with probability ρ. Balanced accuracy for healthy-vs-unhealthy:

| Field-quality photos + … | ρ = 1.0 | ρ = 0.8 | ρ = 0.6 |
| --- | --- | --- | --- |
| Image only | 65.3% | 65.3% | 65.3% |
| + soil-test kit | **80.6%** | **73.4%** | **67.0%** |
| + all sensors | 79.5% | 72.3% | 66.2% |
| + soil moisture only | 70.4% | 67.6% | 65.3% |
| + air/humidity/light only | 65.3% | 65.3% | 65.3% (ignored: no moisture/N) |

With clean lab photos the image model is already ~100% and fusion keeps it there. Full results, including
a soil-weight sensitivity sweep, are in `ml/outputs/fusion_metrics.json` and served at `GET /api/meta`.
**These figures depend on the pairing assumption; real paired field data is needed to confirm them.**

## Run it locally

Requires Python 3.12 and Node 20+.

```bash
# Backend API  ->  http://localhost:8000  (interactive docs at /docs)
python -m venv .venv && source .venv/bin/activate
pip install -r backend/requirements-dev.txt
cd backend && uvicorn app.main:app --reload --port 8000

# Tests (unit + integration against the real models)
cd backend && pytest --cov=app

# Frontend  ->  http://localhost:5173
cd frontend && npm install && npm run dev
```

## API

| Method | Path | Description |
| --- | --- | --- |
| `POST` | `/api/assess` | multipart form: `image` (file, optional), `soil` (JSON string, optional), `crop` (`pepper`/`potato`/`tomato`, optional). At least one of image/soil is required. |
| `GET` | `/api/meta` | Classes, disease info, soil feature definitions & healthy ranges, model metrics |
| `GET` | `/api/samples` | Demo samples (held-out test images + real soil rows) |
| `GET` | `/api/health` | Liveness and model availability |

```bash
curl -X POST http://localhost:8000/api/assess \
  -F image=@leaf.jpg \
  -F 'soil={"soil_moisture": 14.2, "nitrogen": 18.5, "soil_ph": 6.4}'
```

Soil fields (all optional, any subset): `soil_moisture` (%), `soil_ph`, `nitrogen`, `phosphorus`,
`potassium` (mg/kg), `soil_temperature`, `ambient_temperature` (°C), `humidity` (%),
`light_intensity`, `chlorophyll_content`, `electrochemical_signal`. Units follow the source dataset,
which does not document them precisely. Readings outside the training range are accepted but flagged.

Configuration (environment variables): `CORS_ORIGINS` (comma-separated), `CORS_ORIGIN_REGEX`,
`MAX_UPLOAD_MB` (default 4; Vercel caps request bodies at 4.5 MB), `RATE_LIMIT_PER_MINUTE` (default 30 per client, per instance - best-effort; add a Vercel Firewall rate-limit rule for hard limits), `MODELS_DIR`, `SAMPLES_DIR`.

## Retrain the models

The image dataset (PlantVillage, ~735 MB) is not in git. Download PlantVillage and place the class folders
in `Image Data/PlantVillage/` (e.g. `Image Data/PlantVillage/Tomato_healthy/…`). Then:

```bash
pip install -r ml/requirements.txt
python ml/train_image.py      # ~8 min on an Apple-silicon GPU; writes backend/models/leaf_disease.*
python ml/train_soil.py       # seconds; writes backend/models/soil_stress.*
python ml/evaluate_fusion.py  # writes ml/outputs/fusion_metrics.json (+ copy in backend/models)
python ml/build_samples.py    # writes backend/samples/
```

All scripts are seeded. The train/val/test split is saved to `ml/outputs/image_split.csv`.

## Limitations

- Recognises only the 15 PlantVillage classes (pepper, potato, tomato). Other plants get a low-confidence warning but still receive a best-guess label.
- Lab-photo training data: real field photos will score lower than the headline accuracy.
- The soil model is trained on a small, likely synthetic dataset; soil stress ≠ disease.
- Recommendations summarise general extension guidance - this is decision support, not a substitute for a local agronomist.
