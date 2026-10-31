"""Backwards-compatible loss API.

The implementation now lives in :mod:`pid_env.evaluation` so CMA-ES and
dataset generation can share the same rollout code.
"""

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

__all__ = [
    "TRAJECTORY_FEATURES",
    "LossHyperparameters",
    "PIDSettings",
    "RolloutResult",
    "Trajectory",
    "evaluate",
    "pid_loss",
    "rollout",
]
