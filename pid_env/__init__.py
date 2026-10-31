"""1D simulation environment for PID tuning experiments."""

from .config import EnvironmentConfig, NoiseConfig
from .controller import PIDController
from .env import PID1DEnv
from .evaluation import (
    TRAJECTORY_FEATURES,
    LossHyperparameters,
    PIDSettings,
    RolloutResult,
    Trajectory,
    evaluate,
    pid_loss,
    rollout,
)
from .noise import FractalPerlinNoise
from .plotting import plot_trajectory

__all__ = [
    "TRAJECTORY_FEATURES",
    "EnvironmentConfig",
    "LossHyperparameters",
    "NoiseConfig",
    "PIDController",
    "PID1DEnv",
    "PIDSettings",
    "RolloutResult",
    "FractalPerlinNoise",
    "Trajectory",
    "evaluate",
    "pid_loss",
    "plot_trajectory",
    "rollout",
]
