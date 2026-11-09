"""Integration tests against the real trained models and demo samples."""

import io
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.config import BACKEND_DIR, load_settings
from app.main import create_app

pytestmark = pytest.mark.integration

SAMPLES = json.loads((BACKEND_DIR / "samples" / "samples.json").read_text())
HEALTHY_SOIL = next(s["soil"] for s in SAMPLES if s["id"] == "soil-only-healthy")
STRESSED_SOIL = next(s["soil"] for s in SAMPLES if s["id"] == "soil-only-stressed")


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(create_app())


def sample_image_bytes(sample_id: str) -> bytes:
    sample = next(s for s in SAMPLES if s["id"] == sample_id)
    return Path(BACKEND_DIR / "samples" / "images" / sample["image"]).read_bytes()


def post(client: TestClient, image: bytes | None = None, soil: dict | None = None, crop: str | None = None):
    files = {"image": ("leaf.jpg", image, "image/jpeg")} if image is not None else None
    data = {}
    if soil is not None:
        data["soil"] = json.dumps(soil)
    if crop is not None:
        data["crop"] = crop
    return client.post("/api/assess", files=files, data=data)


def test_health_reports_models_present(client):
    body = client.get("/api/health").json()
    assert body == {"status": "ok", "image_model": True, "soil_model": True, "version": "1.0.0"}


def test_meta_lists_classes_features_and_metrics(client):
    body = client.get("/api/meta").json()
    assert len(body["classes"]) == 15
    assert {f["key"] for f in body["soil_features"]} >= {"soil_moisture", "nitrogen", "soil_ph"}
    assert body["metrics"]["image"]["test_accuracy"] > 0.95
    assert "conditions" in body["metrics"]["fusion"]


def test_samples_are_listed_and_images_served(client):
    samples = client.get("/api/samples").json()
    assert len(samples) == len(SAMPLES)
    with_image = next(s for s in samples if s["image_url"])
    response = client.get(with_image["image_url"])
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"


@pytest.mark.parametrize("name", ["../main.py", "..%2Fmain.py", "missing.jpg"])
def test_sample_image_rejects_traversal_and_missing(client, name):
    assert client.get(f"/api/samples/images/{name}").status_code == 404


@pytest.mark.parametrize("sample", [s for s in SAMPLES if s["image"]], ids=lambda s: s["id"])
def test_every_image_sample_is_classified_as_expected(client, sample):
    response = post(client, image=sample_image_bytes(sample["id"]))
    assert response.status_code == 200
    assert response.json()["image"]["prediction"]["class_id"] == sample["expected_label"]


def test_image_only_diseased_leaf(client):
    body = post(client, image=sample_image_bytes("tomato-late-blight")).json()
    assert body["assessment"]["sources"] == ["image"]
    assert body["assessment"]["status"] == "unhealthy"
    assert body["soil"] is None
    assert body["image"]["disease"]["pathogen"] == "Phytophthora infestans"
    assert any("fungicide" in r for r in body["recommendations"])


def test_soil_only_healthy_and_stressed(client):
    healthy = post(client, soil=HEALTHY_SOIL).json()
    stressed = post(client, soil=STRESSED_SOIL).json()
    assert healthy["soil"]["status"] == "Healthy"
    assert stressed["soil"]["status"] == "High Stress"
    assert healthy["assessment"]["health_score"] > stressed["assessment"]["health_score"]
    assert stressed["soil"]["drivers"], "stressed soil should be explained"


def test_combined_inputs_agree_and_strengthen_assessment(client):
    image = sample_image_bytes("tomato-late-blight")
    image_only = post(client, image=image).json()["assessment"]
    combined = post(client, image=image, soil=STRESSED_SOIL).json()["assessment"]
    assert combined["sources"] == ["image", "soil"]
    assert combined["agreement"] == "agree"
    assert combined["health_score"] <= image_only["health_score"]


def test_healthy_leaf_with_stressed_soil_is_flagged_as_conflict(client):
    body = post(client, image=sample_image_bytes("healthy-leaf-dry-soil"), soil=STRESSED_SOIL).json()
    assert body["image"]["prediction"]["class_id"] == "pepper_healthy"
    assert body["assessment"]["agreement"] == "conflict"
    assert body["assessment"]["status"] == "at_risk"
    assert body["assessment"]["headline"].startswith("Leaf looks healthy")
    assert any("irrigate" in r.lower() for r in body["recommendations"])


def test_healthy_leaf_with_healthy_soil_is_healthy(client):
    body = post(client, image=sample_image_bytes("healthy-tomato"), soil=HEALTHY_SOIL).json()
    assert body["assessment"]["status"] == "healthy"
    assert body["assessment"]["agreement"] == "agree"


def test_crop_hint_restricts_predictions(client):
    body = post(client, image=sample_image_bytes("tomato-late-blight"), crop="potato").json()
    assert all(p["crop"] == "potato" for p in body["image"]["top_predictions"])


def test_partial_soil_without_key_features_warns(client):
    body = post(client, soil={"humidity": 55, "ambient_temperature": 24}).json()
    assert body["soil"]["informative"] is False
    assert any("not used in the combined score" in w for w in body["warnings"])
    assert body["assessment"]["contributions"]["soil"] == 0


def test_out_of_training_range_readings_warn(client):
    body = post(client, soil={"soil_moisture": 80, "nitrogen": 30}).json()
    assert "soil_moisture" in body["soil"]["out_of_training_range"]


def test_crop_specific_ph_advice(client):
    body = post(client, image=sample_image_bytes("healthy-tomato"), soil={"soil_ph": 5.0, "soil_moisture": 35}).json()
    assert any("pH 5 is below" in r for r in body["recommendations"])


def test_requires_some_input(client):
    response = post(client)
    assert response.status_code == 400
    assert "Provide a leaf image" in response.json()["detail"]


@pytest.mark.parametrize(
    "soil_raw", ['{"soil_ph": 20}', '{"unknown": 1}', "not json", "{}"]
)
def test_invalid_soil_is_rejected(client, soil_raw):
    response = client.post("/api/assess", data={"soil": soil_raw})
    assert response.status_code == 422
    assert "Invalid soil data" in response.json()["detail"]


def test_invalid_crop_is_rejected(client):
    assert post(client, soil=HEALTHY_SOIL, crop="banana").status_code == 422


def test_non_image_upload_is_rejected(client):
    response = post(client, image=b"%PDF-1.4 definitely not an image")
    assert response.status_code == 400
    assert "could not be read" in response.json()["detail"]


def test_oversized_upload_is_rejected(monkeypatch):
    monkeypatch.setenv("MAX_UPLOAD_MB", "0.01")
    small_client = TestClient(create_app(load_settings()))
    buffer = io.BytesIO()
    Image.effect_noise((300, 300), 80).convert("RGB").save(buffer, format="PNG")
    response = post(small_client, image=buffer.getvalue())
    assert response.status_code == 413


def test_cors_allows_configured_origin(client):
    response = client.options(
        "/api/assess",
        headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "POST"},
    )
    assert response.headers.get("access-control-allow-origin") == "http://localhost:5173"


def test_crop_hint_mismatch_is_warned(client):
    body = post(client, image=sample_image_bytes("tomato-late-blight"), crop="pepper").json()
    assert body["image"]["crop_mismatch"] is True
    assert body["image"]["predicted_crop"] == "tomato"
    assert any("looks more like a tomato" in w for w in body["warnings"])


def test_extreme_aspect_ratio_is_rejected_quickly(client):
    buffer = io.BytesIO()
    Image.new("RGB", (400_000, 10)).save(buffer, format="PNG")
    response = post(client, image=buffer.getvalue())
    assert response.status_code == 400
    assert "long and thin" in response.json()["detail"]


def test_unsupported_format_is_rejected(client):
    buffer = io.BytesIO()
    Image.new("RGB", (64, 64)).save(buffer, format="TIFF")
    assert post(client, image=buffer.getvalue()).status_code == 400


def test_oversized_request_is_rejected_before_parsing(monkeypatch):
    monkeypatch.setenv("MAX_UPLOAD_MB", "0.01")
    small_client = TestClient(create_app(load_settings()))
    response = small_client.post(
        "/api/assess",
        content=b"x" * 200_000,
        headers={"content-type": "multipart/form-data; boundary=zzz"},
    )
    assert response.status_code == 413


def test_rate_limit_ignores_spoofed_forwarded_for(monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "2")
    limited = TestClient(create_app(load_settings()))
    codes = [
        post_with_headers(limited, {"x-forwarded-for": f"10.0.0.{i}"}).status_code for i in range(4)
    ]
    assert codes[:2] == [200, 200]
    assert codes[2:] == [429, 429]


def test_rate_limit_uses_platform_client_ip(monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "1")
    limited = TestClient(create_app(load_settings()))
    first = post_with_headers(limited, {"x-real-ip": "1.1.1.1"})
    second = post_with_headers(limited, {"x-real-ip": "2.2.2.2"})
    repeat = post_with_headers(limited, {"x-real-ip": "1.1.1.1"})
    assert (first.status_code, second.status_code, repeat.status_code) == (200, 200, 429)


def post_with_headers(test_client: TestClient, headers: dict):
    return test_client.post("/api/assess", data={"soil": json.dumps(HEALTHY_SOIL)}, headers=headers)


def test_multiple_bad_readings_are_all_explained(client):
    body = post(client, soil={"soil_moisture": 14, "nitrogen": 15}).json()
    assert body["assessment"]["status"] == "unhealthy"
    drivers = {d["feature"] for d in body["soil"]["drivers"]}
    assert drivers == {"soil_moisture", "nitrogen"}
    assert not any("No action needed" in r for r in body["recommendations"])


def test_driver_messages_use_rounded_ranges(client):
    body = post(client, soil={"soil_moisture": 14, "nitrogen": 15}).json()
    message = next(d["message"] for d in body["soil"]["drivers"] if d["feature"] == "soil_moisture")
    assert "30.92-39.26" in message


def test_unhealthy_result_never_says_no_action_needed(client):
    for soil in ({"soil_moisture": 12}, {"nitrogen": 12}, STRESSED_SOIL):
        body = post(client, soil=soil).json()
        if body["assessment"]["status"] != "healthy":
            assert not any("No action needed" in r for r in body["recommendations"])
