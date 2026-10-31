"""Configuration and result dataclasses for tuning and dataset generation."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..config import EnvironmentConfig
from ..evaluation import LossHyperparameters, PIDSettings


def _positive_range(name: str, value: tuple[float, float]) -> None:
    if len(value) != 2 or value[0] <= 0.0 or value[0] > value[1]:
        raise ValueError(f"{name} must be a positive (low, high) tuple")


@dataclass
class GainSamplerConfig:
    """Distribution used to sample the controller's current gains.

    Randomising prior gains for every training sample prevents the model from
    overfitting to one starting controller.
    """

    kp_range: tuple[float, float] = (0.05, 10.0)
    ki_range: tuple[float, float] = (0.01, 2.0)
    kd_range: tuple[float, float] = (0.01, 5.0)
    zero_ki_probability: float = 0.2
    zero_kd_probability: float = 0.2
    log_uniform: bool = True

    def __post_init__(self) -> None:
        _positive_range("kp_range", self.kp_range)
        _positive_range("ki_range", self.ki_range)
        _positive_range("kd_range", self.kd_range)
        if not 0.0 <= self.zero_ki_probability <= 1.0:
            raise ValueError("zero_ki_probability must be in [0, 1]")
        if not 0.0 <= self.zero_kd_probability <= 1.0:
            raise ValueError("zero_kd_probability must be in [0, 1]")


@dataclass
class EnvSamplerConfig:
    """Ranges used to generate arbitrary environment configurations."""

    mass_range: tuple[float, float] = (0.5, 2.0)
    weight_range: tuple[float, float] = (5.0, 20.0)
    drag_range: tuple[float, float] = (0.01, 2.0)
    thrust_range: tuple[float, float] = (10.0, 60.0)
    target_range: tuple[float, float] = (0.25, 2.0)
    initial_position_range: tuple[float, float] = (-0.5, 0.5)
    initial_velocity_range: tuple[float, float] = (-0.5, 0.5)
    disturbance_strength_range: tuple[float, float] = (0.0, 8.0)
    disturbance_frequency_range: tuple[float, float] = (0.3, 3.0)
    target_noise_strength_range: tuple[float, float] = (0.0, 0.5)
    target_noise_frequency_range: tuple[float, float] = (0.1, 1.0)
    dt_choices: tuple[float, ...] = (0.01, 0.02, 0.05)
    noise_octaves_range: tuple[int, int] = (2, 6)
    noise_persistence_range: tuple[float, float] = (0.35, 0.65)
    noise_lacunarity_range: tuple[float, float] = (1.5, 2.5)
    noise_frequency_range: tuple[float, float] = (0.5, 2.0)
    bounds_margin_range: tuple[float, float] = (2.0, 5.0)
    feasibility_margin: float = 1.15
    feasible_only: bool = True
    max_attempts: int = 100

    def __post_init__(self) -> None:
        for name in (
            "mass_range",
            "weight_range",
            "drag_range",
            "thrust_range",
            "target_range",
            "disturbance_frequency_range",
            "target_noise_frequency_range",
            "noise_lacunarity_range",
            "noise_frequency_range",
            "bounds_margin_range",
        ):
            _positive_range(name, getattr(self, name))
        _positive_range("noise_persistence_range", self.noise_persistence_range)
        if self.noise_persistence_range[1] > 1.0:
            raise ValueError("noise_persistence_range must be <= 1")
        if self.noise_octaves_range[0] < 1:
            raise ValueError("noise_octaves_range low must be >= 1")
        if self.noise_octaves_range[0] > self.noise_octaves_range[1]:
            raise ValueError("noise_octaves_range low must be <= high")
        if not self.dt_choices or any(dt <= 0.0 for dt in self.dt_choices):
            raise ValueError("dt_choices must contain positive values")
        if self.feasibility_margin < 1.0:
            raise ValueError("feasibility_margin must be >= 1")


@dataclass
class ProbeConfig:
    """Settings for the closed-loop trajectory used as transformer input."""

    target_offsets: tuple[float, ...] = (0.0, 1.0, -0.5, 0.5, 0.0)
    segment_steps: int = 100
    priors_per_config: int = 4
    probe_seeds: tuple[int, ...] = (0,)
    max_prior_attempts: int = 5

    def __post_init__(self) -> None:
        if not self.target_offsets:
            raise ValueError("target_offsets must not be empty")
        if self.segment_steps < 1:
            raise ValueError("segment_steps must be >= 1")
        if self.priors_per_config < 1:
            raise ValueError("priors_per_config must be >= 1")
        if not self.probe_seeds:
            raise ValueError("probe_seeds must not be empty")
        if self.max_prior_attempts < 1:
            raise ValueError("max_prior_attempts must be >= 1")


@dataclass
class CMAESConfig:
    """CMA-ES search settings for a single environment configuration."""

    initial_gains: PIDSettings = field(
        default_factory=lambda: PIDSettings(kp=1.0, ki=0.1, kd=0.1)
    )
    sigma0: float = 0.5
    popsize: int | None = None
    max_evaluations: int = 800
    seeds: tuple[int, ...] = (0, 1, 2)
    validation_seeds: tuple[int, ...] = (1_000, 1_001)
    seed: int = 0
    restarts: int = 0
    workers: int = 1
    gain_bounds: tuple[float, float] = (1e-6, 1e3)
    log_space: bool = True
    tolfun: float = 1e-9
    verbose: int = -9

    def __post_init__(self) -> None:
        if self.sigma0 <= 0.0:
            raise ValueError("sigma0 must be > 0")
        if self.popsize is not None and self.popsize < 2:
            raise ValueError("popsize must be >= 2")
        if self.max_evaluations < 1:
            raise ValueError("max_evaluations must be >= 1")
        if not self.seeds:
            raise ValueError("seeds must not be empty")
        if not self.validation_seeds:
            raise ValueError("validation_seeds must not be empty")
        if self.restarts < 0:
            raise ValueError("restarts must be >= 0")
        if self.workers < 1:
            raise ValueError("workers must be >= 1")
        low, high = self.gain_bounds
        if low <= 0.0 or low >= high:
            raise ValueError("gain_bounds must be positive (low, high)")


@dataclass
class CMAESResult:
    """Best result found by :func:`pid_env.tuning.cmaes.tune`."""

    best_gains: PIDSettings
    best_loss: float
    validation_loss: float
    evaluations: int
    generations: int
    history: list[dict]
    env_config: EnvironmentConfig
    hyperparameters: LossHyperparameters
    cma_config: CMAESConfig

    def to_dict(self) -> dict:
        from .serialization import (
            cmaes_config_to_dict,
            environment_config_to_dict,
            loss_hyperparameters_to_dict,
            pid_settings_to_dict,
        )

        return {
            "best_gains": pid_settings_to_dict(self.best_gains),
            "best_loss": self.best_loss,
            "validation_loss": self.validation_loss,
            "evaluations": self.evaluations,
            "generations": self.generations,
            "history": self.history,
            "env_config": environment_config_to_dict(self.env_config),
            "hyperparameters": loss_hyperparameters_to_dict(self.hyperparameters),
            "cma_config": cmaes_config_to_dict(self.cma_config),
        }


@dataclass
class DatasetConfig:
    """Top-level dataset-generation settings."""

    output_dir: str = "data/pid_dataset"
    name: str = "pid-transformer"
    num_configs: int = 1_000
    shard_size: int = 128
    horizon_steps: int = 600
    seed: int = 0
    workers: int = 1
    resume: bool = True
    train_fraction: float = 0.8
    validation_fraction: float = 0.1
    test_fraction: float = 0.1

    def __post_init__(self) -> None:
        if self.num_configs < 1:
            raise ValueError("num_configs must be >= 1")
        if self.shard_size < 1:
            raise ValueError("shard_size must be >= 1")
        if self.horizon_steps < 1:
            raise ValueError("horizon_steps must be >= 1")
        if self.workers < 1:
            raise ValueError("workers must be >= 1")
        total = (
            self.train_fraction + self.validation_fraction + self.test_fraction
        )
        if abs(total - 1.0) > 1e-6:
            raise ValueError("split fractions must sum to 1")
