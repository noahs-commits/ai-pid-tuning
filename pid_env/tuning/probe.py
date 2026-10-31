"""Closed-loop probe trajectories used as transformer inputs."""

from __future__ import annotations

import numpy as np

from ..config import EnvironmentConfig
from ..evaluation import (
    LossHyperparameters,
    PIDSettings,
    RolloutResult,
    rollout,
)
from .sampler import sample_prior_gains
from .types import GainSamplerConfig, ProbeConfig


def probe_rollout(
    env_config: EnvironmentConfig,
    prior_gains: PIDSettings,
    hyperparameters: LossHyperparameters,
    probe: ProbeConfig,
    *,
    seed: int,
) -> RolloutResult:
    """Run one closed-loop probe under ``prior_gains``."""
    return rollout(
        prior_gains,
        env_config,
        hyperparameters,
        seed=seed,
        record=True,
        target_offsets=probe.target_offsets,
        segment_steps=probe.segment_steps,
    )


def sample_probe(
    env_config: EnvironmentConfig,
    hyperparameters: LossHyperparameters,
    probe: ProbeConfig,
    gain_sampler: GainSamplerConfig | None,
    rng: np.random.Generator,
    *,
    seed: int,
) -> tuple[RolloutResult, PIDSettings]:
    """Sample a fresh prior controller and record its closed-loop response.

    Prior gains are drawn anew for every sample.  If the resulting probe
    terminates early, the draw is retried a few times so the dataset does not
    fill up with unusable crashers.
    """
    last_result: RolloutResult | None = None
    last_gains: PIDSettings | None = None

    for _ in range(probe.max_prior_attempts):
        prior_gains = sample_prior_gains(rng, gain_sampler)
        result = probe_rollout(
            env_config,
            prior_gains,
            hyperparameters,
            probe,
            seed=seed,
        )
        last_result = result
        last_gains = prior_gains
        if not result.terminated:
            return result, prior_gains

    assert last_result is not None and last_gains is not None
    return last_result, last_gains
