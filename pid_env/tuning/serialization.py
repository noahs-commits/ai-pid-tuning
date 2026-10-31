"""JSON-friendly conversion helpers for tuning configurations."""

from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path
from typing import Any, TypeVar

from ..config import EnvironmentConfig, NoiseConfig
from ..evaluation import LossHyperparameters, PIDSettings
from .types import (
    CMAESConfig,
    DatasetConfig,
    EnvSamplerConfig,
    GainSamplerConfig,
    ProbeConfig,
)

T = TypeVar("T")


def _filtered(cls: type[T], data: dict[str, Any]) -> dict[str, Any]:
    names = {field.name for field in fields(cls)}
    return {key: value for key, value in data.items() if key in names}


def noise_config_to_dict(config: NoiseConfig) -> dict[str, Any]:
    return {
        "seed": config.seed,
        "octaves": config.octaves,
        "persistence": config.persistence,
        "lacunarity": config.lacunarity,
        "frequency": config.frequency,
    }


def noise_config_from_dict(data: dict[str, Any]) -> NoiseConfig:
    return NoiseConfig(**_filtered(NoiseConfig, data))


def environment_config_to_dict(config: EnvironmentConfig) -> dict[str, Any]:
    return {
        "dt": config.dt,
        "max_steps": config.max_steps,
        "mass": config.mass,
        "weight": config.weight,
        "drag": config.drag,
        "thrust": config.thrust,
        "target": config.target,
        "initial_position": config.initial_position,
        "initial_velocity": config.initial_velocity,
        "disturbance_strength": config.disturbance_strength,
        "disturbance_frequency": config.disturbance_frequency,
        "disturbance_noise": noise_config_to_dict(config.disturbance_noise),
        "target_noise_strength": config.target_noise_strength,
        "target_noise_frequency": config.target_noise_frequency,
        "target_noise": noise_config_to_dict(config.target_noise),
        "bounds": list(config.bounds) if config.bounds is not None else None,
        "reward_control_cost": config.reward_control_cost,
    }


def environment_config_from_dict(data: dict[str, Any]) -> EnvironmentConfig:
    values = _filtered(EnvironmentConfig, data)
    for key in ("disturbance_noise", "target_noise"):
        if key in values:
            values[key] = noise_config_from_dict(values[key])
    if values.get("bounds") is not None:
        values["bounds"] = tuple(values["bounds"])
    return EnvironmentConfig(**values)


def pid_settings_to_dict(settings: PIDSettings) -> dict[str, Any]:
    return {
        "kp": settings.kp,
        "ki": settings.ki,
        "kd": settings.kd,
        "integral_limits": (
            list(settings.integral_limits)
            if settings.integral_limits is not None
            else None
        ),
    }


def pid_settings_from_dict(data: dict[str, Any]) -> PIDSettings:
    values = _filtered(PIDSettings, data)
    if values.get("integral_limits") is not None:
        values["integral_limits"] = tuple(values["integral_limits"])
    return PIDSettings(**values)


def loss_hyperparameters_to_dict(
    hyperparameters: LossHyperparameters,
) -> dict[str, float]:
    return {
        "position_error_weight": hyperparameters.position_error_weight,
        "velocity_weight": hyperparameters.velocity_weight,
        "acceleration_weight": hyperparameters.acceleration_weight,
    }


def loss_hyperparameters_from_dict(
    data: dict[str, Any],
) -> LossHyperparameters:
    return LossHyperparameters(**_filtered(LossHyperparameters, data))


def gain_sampler_to_dict(config: GainSamplerConfig) -> dict[str, Any]:
    return {
        "kp_range": list(config.kp_range),
        "ki_range": list(config.ki_range),
        "kd_range": list(config.kd_range),
        "zero_ki_probability": config.zero_ki_probability,
        "zero_kd_probability": config.zero_kd_probability,
        "log_uniform": config.log_uniform,
    }


def gain_sampler_from_dict(data: dict[str, Any]) -> GainSamplerConfig:
    values = _filtered(GainSamplerConfig, data)
    for key in ("kp_range", "ki_range", "kd_range"):
        if key in values:
            values[key] = tuple(values[key])
    return GainSamplerConfig(**values)


def env_sampler_to_dict(config: EnvSamplerConfig) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for field in fields(EnvSamplerConfig):
        value = getattr(config, field.name)
        values[field.name] = list(value) if isinstance(value, tuple) else value
    return values


def env_sampler_from_dict(data: dict[str, Any]) -> EnvSamplerConfig:
    values = _filtered(EnvSamplerConfig, data)
    for key, value in list(values.items()):
        if isinstance(value, list):
            values[key] = tuple(value)
    return EnvSamplerConfig(**values)


def probe_config_to_dict(config: ProbeConfig) -> dict[str, Any]:
    return {
        "target_offsets": list(config.target_offsets),
        "segment_steps": config.segment_steps,
        "priors_per_config": config.priors_per_config,
        "probe_seeds": list(config.probe_seeds),
        "max_prior_attempts": config.max_prior_attempts,
    }


def probe_config_from_dict(data: dict[str, Any]) -> ProbeConfig:
    values = _filtered(ProbeConfig, data)
    for key in ("target_offsets", "probe_seeds"):
        if key in values:
            values[key] = tuple(values[key])
    return ProbeConfig(**values)


def cmaes_config_to_dict(config: CMAESConfig) -> dict[str, Any]:
    return {
        "initial_gains": pid_settings_to_dict(config.initial_gains),
        "sigma0": config.sigma0,
        "popsize": config.popsize,
        "max_evaluations": config.max_evaluations,
        "seeds": list(config.seeds),
        "validation_seeds": list(config.validation_seeds),
        "seed": config.seed,
        "restarts": config.restarts,
        "workers": config.workers,
        "gain_bounds": list(config.gain_bounds),
        "log_space": config.log_space,
        "tolfun": config.tolfun,
        "verbose": config.verbose,
    }


def cmaes_config_from_dict(data: dict[str, Any]) -> CMAESConfig:
    values = _filtered(CMAESConfig, data)
    if "initial_gains" in values:
        values["initial_gains"] = pid_settings_from_dict(values["initial_gains"])
    for key in ("seeds", "validation_seeds", "gain_bounds"):
        if key in values:
            values[key] = tuple(values[key])
    return CMAESConfig(**values)


def dataset_config_to_dict(config: DatasetConfig) -> dict[str, Any]:
    return {field.name: getattr(config, field.name) for field in fields(DatasetConfig)}


def dataset_config_from_dict(data: dict[str, Any]) -> DatasetConfig:
    return DatasetConfig(**_filtered(DatasetConfig, data))


def to_jsonable(value: Any) -> Any:
    """Recursively convert dataclasses, tuples and numpy values to JSON types."""
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {key: to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value


def save_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(to_jsonable(payload), handle, indent=2, sort_keys=True)


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)
