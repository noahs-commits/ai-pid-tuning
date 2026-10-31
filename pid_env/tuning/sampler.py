"""Random environment and prior-gain sampling for dataset generation."""

from __future__ import annotations

import numpy as np

from ..config import EnvironmentConfig, NoiseConfig
from ..evaluation import PIDSettings
from .types import EnvSamplerConfig, GainSamplerConfig

ENV_FEATURE_NAMES = (
    "mass",
    "weight",
    "drag",
    "thrust",
    "target",
    "dt",
    "max_steps",
    "initial_position",
    "initial_velocity",
    "disturbance_strength",
    "disturbance_frequency",
    "target_noise_strength",
    "target_noise_frequency",
    "bounds_low",
    "bounds_high",
    "disturbance_octaves",
    "disturbance_persistence",
    "disturbance_lacunarity",
    "disturbance_noise_frequency",
    "target_octaves",
    "target_persistence",
    "target_lacunarity",
    "target_noise_field_frequency",
)


def _log_uniform(
    rng: np.random.Generator, low: float, high: float
) -> float:
    return float(np.exp(rng.uniform(np.log(low), np.log(high))))


def _sample_range(
    rng: np.random.Generator,
    bounds: tuple[float, float],
    *,
    log: bool = False,
) -> float:
    low, high = bounds
    if log:
        return _log_uniform(rng, low, high)
    return float(rng.uniform(low, high))


def _sample_noise(rng: np.random.Generator, config: EnvSamplerConfig) -> NoiseConfig:
    return NoiseConfig(
        seed=int(rng.integers(0, 2**31 - 1)),
        octaves=int(
            rng.integers(
                config.noise_octaves_range[0],
                config.noise_octaves_range[1] + 1,
            )
        ),
        persistence=_sample_range(rng, config.noise_persistence_range),
        lacunarity=_sample_range(
            rng, config.noise_lacunarity_range, log=True
        ),
        frequency=_sample_range(rng, config.noise_frequency_range, log=True),
    )


def sample_environment(
    rng: np.random.Generator,
    config: EnvSamplerConfig | None = None,
    *,
    horizon_steps: int = 600,
) -> EnvironmentConfig:
    """Sample a feasible :class:`EnvironmentConfig`."""
    config = config or EnvSamplerConfig()

    for _ in range(config.max_attempts):
        mass = _sample_range(rng, config.mass_range, log=True)
        weight = _sample_range(rng, config.weight_range, log=True)
        drag = _sample_range(rng, config.drag_range, log=True)
        thrust = _sample_range(rng, config.thrust_range, log=True)
        if config.feasible_only and thrust < weight * config.feasibility_margin:
            continue

        target = _sample_range(rng, config.target_range, log=False)
        initial_position = _sample_range(rng, config.initial_position_range)
        initial_velocity = _sample_range(rng, config.initial_velocity_range)

        disturbance_strength = _sample_range(
            rng, config.disturbance_strength_range
        )
        disturbance_frequency = _sample_range(
            rng, config.disturbance_frequency_range, log=True
        )
        target_noise_strength = _sample_range(
            rng, config.target_noise_strength_range
        )
        target_noise_frequency = _sample_range(
            rng, config.target_noise_frequency_range, log=True
        )

        margin = _sample_range(rng, config.bounds_margin_range, log=True)
        reach = target_noise_strength + margin
        low = min(initial_position, target) - reach
        high = max(initial_position, target) + reach
        if low >= high:
            continue

        dt = float(rng.choice(config.dt_choices))
        return EnvironmentConfig(
            dt=dt,
            max_steps=int(horizon_steps),
            mass=mass,
            weight=weight,
            drag=drag,
            thrust=thrust,
            target=target,
            initial_position=initial_position,
            initial_velocity=initial_velocity,
            disturbance_strength=disturbance_strength,
            disturbance_frequency=disturbance_frequency,
            disturbance_noise=_sample_noise(rng, config),
            target_noise_strength=target_noise_strength,
            target_noise_frequency=target_noise_frequency,
            target_noise=_sample_noise(rng, config),
            bounds=(low, high),
            reward_control_cost=0.0,
        )

    raise RuntimeError(
        f"could not sample a feasible environment after {config.max_attempts} attempts"
    )


def sample_prior_gains(
    rng: np.random.Generator,
    config: GainSamplerConfig | None = None,
) -> PIDSettings:
    """Sample a random current controller, including sometimes zero I/D."""
    config = config or GainSamplerConfig()

    def draw(bounds: tuple[float, float], zero_probability: float) -> float:
        if zero_probability > 0.0 and rng.random() < zero_probability:
            return 0.0
        return _sample_range(rng, bounds, log=config.log_uniform)

    return PIDSettings(
        kp=draw(config.kp_range, 0.0),
        ki=draw(config.ki_range, config.zero_ki_probability),
        kd=draw(config.kd_range, config.zero_kd_probability),
        integral_limits=None,
    )


def env_feature_vector(config: EnvironmentConfig) -> np.ndarray:
    """Return numeric environment features in :data:`ENV_FEATURE_NAMES` order."""
    bounds = config.bounds
    values = [
        config.mass,
        config.weight,
        config.drag,
        config.thrust,
        config.target,
        config.dt,
        config.max_steps,
        config.initial_position,
        config.initial_velocity,
        config.disturbance_strength,
        config.disturbance_frequency,
        config.target_noise_strength,
        config.target_noise_frequency,
        bounds[0] if bounds is not None else np.nan,
        bounds[1] if bounds is not None else np.nan,
        config.disturbance_noise.octaves,
        config.disturbance_noise.persistence,
        config.disturbance_noise.lacunarity,
        config.disturbance_noise.frequency,
        config.target_noise.octaves,
        config.target_noise.persistence,
        config.target_noise.lacunarity,
        config.target_noise.frequency,
    ]
    return np.asarray(values, dtype=np.float32)
