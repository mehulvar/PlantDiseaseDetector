"""Combine leaf-image and soil-sensor evidence into one plant-health assessment.

Method: Bayesian evidence fusion in log-odds space.

Each model outputs P(healthy). Because each model was trained on data with its
own class balance, its raw probability mixes *evidence* with that training
prior. We remove the prior to obtain a log-likelihood ratio (the evidence),
weight it by how directly that modality observes plant health, then add the
evidence terms to a neutral 50/50 prior:

    logit(P_fused) = sum_m  w_m * (logit(p_m) - logit(prior_m))

Properties this guarantees:
* An uninformative reading (p_m == prior_m) contributes exactly zero.
* With one modality, the result is that model's prior-corrected opinion.
* Agreeing modalities reinforce each other; conflicting ones moderate the result.

Two safeguards keep any single source from silencing the other:
* Image probabilities are floored at 2% / 98%. The model scores ~99.9% on lab
  photos but is far less reliable on degraded field photos, so a single photo
  should never count as near-certain evidence.
* A plant is only called "healthy" if no source shows strong evidence of a
  problem. A healthy-looking leaf in parched, nutrient-poor soil is reported as
  "at risk" (score capped just below the healthy threshold) - an early warning.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

PROBABILITY_FLOOR = 1e-3
IMAGE_PROBABILITY_FLOOR = 0.02
HEALTHY_THRESHOLD = 0.70
UNHEALTHY_THRESHOLD = 0.35
# Evidence weaker than this (|log-likelihood ratio|) is treated as neutral when
# deciding whether sources agree; logit(0.62) ~= 0.5.
NEUTRAL_EVIDENCE = 0.5
# Weighted evidence at or below this marks a source as reporting a clear problem.
STRONG_NEGATIVE_EVIDENCE = -1.0
AT_RISK_CAP = HEALTHY_THRESHOLD - 0.001
# The image model's calibrated probabilities are used as-is (neutral prior).
IMAGE_PRIOR = 0.5

# How directly each modality observes plant health. Images observe disease
# symptoms directly; soil readings measure stress, an indirect risk factor.
IMAGE_WEIGHT = 1.0
SOIL_WEIGHT = 0.5


@dataclass(frozen=True)
class Evidence:
    """One modality's opinion: P(healthy) and the healthy share in its training data."""

    source: str
    p_healthy: float
    training_prior: float
    weight: float
    floor: float = PROBABILITY_FLOOR

    @property
    def log_likelihood_ratio(self) -> float:
        return logit(self.p_healthy, self.floor) - logit(self.training_prior)

    @property
    def weighted_evidence(self) -> float:
        return self.weight * self.log_likelihood_ratio


@dataclass(frozen=True)
class FusionResult:
    p_healthy: float
    status: str
    agreement: str
    contributions: dict[str, float]


def image_evidence(p_healthy: float) -> Evidence:
    return Evidence("image", p_healthy, IMAGE_PRIOR, IMAGE_WEIGHT, IMAGE_PROBABILITY_FLOOR)


def soil_evidence(p_healthy: float, training_prior: float, weight: float = SOIL_WEIGHT) -> Evidence:
    return Evidence("soil", p_healthy, training_prior, weight)


def clamp_probability(p: float, floor: float = PROBABILITY_FLOOR) -> float:
    return min(max(p, floor), 1.0 - floor)


def logit(p: float, floor: float = PROBABILITY_FLOOR) -> float:
    p = clamp_probability(p, floor)
    return math.log(p / (1.0 - p))


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def status_from_probability(p_healthy: float) -> str:
    if p_healthy >= HEALTHY_THRESHOLD:
        return "healthy"
    if p_healthy <= UNHEALTHY_THRESHOLD:
        return "unhealthy"
    return "at_risk"


def agreement_of(evidence: tuple[Evidence, ...]) -> str:
    """Compare the direction of each decisive source; fewer than two means no comparison."""
    decisive = [
        e for e in evidence if e.weight > 0 and abs(e.log_likelihood_ratio) >= NEUTRAL_EVIDENCE
    ]
    if len(decisive) < 2:
        return "single_source"
    signs = {e.log_likelihood_ratio > 0 for e in decisive}
    return "conflict" if len(signs) > 1 else "agree"


def fuse(evidence: tuple[Evidence, ...]) -> FusionResult:
    if not evidence:
        raise ValueError("At least one source of evidence is required.")
    p_healthy = sigmoid(sum(e.weighted_evidence for e in evidence))
    if any(e.weighted_evidence <= STRONG_NEGATIVE_EVIDENCE for e in evidence):
        p_healthy = min(p_healthy, AT_RISK_CAP)
    # Round once so the displayed score (one decimal of a percent) and the status always agree.
    p_healthy = round(p_healthy, 3)
    return FusionResult(
        p_healthy=p_healthy,
        status=status_from_probability(p_healthy),
        agreement=agreement_of(evidence),
        contributions={e.source: e.weighted_evidence for e in evidence},
    )
