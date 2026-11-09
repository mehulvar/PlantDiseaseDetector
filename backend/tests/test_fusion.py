import math

import pytest

from app.services.fusion import (
    AT_RISK_CAP,
    Evidence,
    fuse,
    image_evidence,
    soil_evidence,
    logit,
    sigmoid,
    status_from_probability,
)

pytestmark = pytest.mark.unit


def test_logit_and_sigmoid_are_inverse():
    for p in (0.01, 0.25, 0.5, 0.9):
        assert sigmoid(logit(p)) == pytest.approx(p)


def test_logit_clamps_extremes_to_finite_values():
    assert math.isfinite(logit(0.0))
    assert math.isfinite(logit(1.0))


def test_uninformative_evidence_contributes_nothing():
    neutral = Evidence("soil", p_healthy=0.25, training_prior=0.25, weight=0.5)
    assert neutral.weighted_evidence == pytest.approx(0.0)
    assert fuse((neutral,)).p_healthy == pytest.approx(0.5)


def test_single_image_source_returns_its_own_probability():
    result = fuse((Evidence("image", 0.9, 0.5, 1.0),))
    assert result.p_healthy == pytest.approx(0.9)
    assert result.agreement == "single_source"


def test_agreeing_sources_reinforce_each_other():
    image = Evidence("image", 0.2, 0.5, 1.0)
    soil = Evidence("soil", 0.05, 0.25, 0.5)
    fused = fuse((image, soil))
    assert fused.p_healthy < 0.2
    assert fused.agreement == "agree"
    assert fused.status == "unhealthy"


def test_conflicting_sources_moderate_the_result():
    image = Evidence("image", 0.9, 0.5, 1.0)
    soil = Evidence("soil", 0.02, 0.25, 0.5)
    fused = fuse((image, soil))
    assert 0.35 < fused.p_healthy < 0.9
    assert fused.agreement == "conflict"


def test_near_neutral_source_does_not_count_as_conflict():
    image = Evidence("image", 0.95, 0.5, 1.0)
    soil = Evidence("soil", 0.22, 0.25, 0.5)  # barely below its prior
    assert fuse((image, soil)).agreement == "agree"


def test_fuse_requires_evidence():
    with pytest.raises(ValueError):
        fuse(())


@pytest.mark.parametrize(
    ("p", "expected"), [(0.9, "healthy"), (0.7, "healthy"), (0.5, "at_risk"), (0.35, "unhealthy"), (0.1, "unhealthy")]
)
def test_status_thresholds(p, expected):
    assert status_from_probability(p) == expected


def test_image_evidence_is_floored_so_one_photo_is_never_certain():
    assert image_evidence(1.0).log_likelihood_ratio == pytest.approx(math.log(0.98 / 0.02))


def test_strong_negative_source_prevents_healthy_status():
    fused = fuse((image_evidence(0.999), soil_evidence(0.001, 0.25)))
    assert fused.status == "at_risk"
    assert fused.p_healthy == pytest.approx(AT_RISK_CAP)
    assert fused.agreement == "conflict"


def test_mild_negative_source_does_not_trigger_safeguard():
    fused = fuse((image_evidence(0.98), soil_evidence(0.15, 0.25)))
    assert fused.status == "healthy"
