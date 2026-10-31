"""Rollout evaluation shared by losses, CMA-ES and dataset generation."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .config import EnvironmentConfig
from .controller import PIDController
from .env import PID1DEnv

TRAJECTORY_FEATURES = ("error", "target", "position", "velocity", "action", "dt")


@dataclass
class PIDSettings:
    """PID gains used when evaluating a loss.

    Kept separate from :class:`PIDController` so optimisers can pass plain
    parameter values without constructing a stateful controller.
    """

    kp: float
    ki: float = 0.0
    kd: float = 0.0
    integral_limits: tuple[float, float] | None = None


@dataclass
class LossHyperparameters:
    """Weights for each penalty term in :func:`pid_loss`.

    All penalties are squared and averaged per simulation step.  Setting a
    weight to zero disables that term.
    """

    position_error_weight: float = 1.0
    velocity_weight: float = 0.0
    acceleration_weight: float = 0.0

    def __post_init__(self) -> None:
        if self.position_error_weight < 0.0:
            raise ValueError("position_error_weight must be >= 0")
        if self.velocity_weight < 0.0:
            raise ValueError("velocity_weight must be >= 0")
        if self.acceleration_weight < 0.0:
            raise ValueError("acceleration_weight must be >= 0")


@dataclass
class Trajectory:
    """Recorded closed-loop signals from one rollout."""

    time: np.ndarray
    position: np.ndarray
    velocity: np.ndarray
    target: np.ndarray
    error: np.ndarray
    action: np.ndarray
    dt: float

    def __len__(self) -> int:
        return int(self.time.size)

    def features(self) -> np.ndarray:
        """Return a ``[T, 6]`` feature matrix using :data:`TRAJECTORY_FEATURES`."""
        dt_column = np.full((len(self), 1), self.dt, dtype=np.float32)
        return np.column_stack(
            [
                self.error,
                self.target,
                self.position,
                self.velocity,
                self.action,
                dt_column,
            ]
        ).astype(np.float32)


@dataclass
class RolloutResult:
    """Outcome of evaluating one PID setting for one episode."""

    loss: float
    terminated: bool
    truncated: bool
    steps: int
    seed: int
    trajectory: Trajectory | None = None


def _make_pid(pid_settings: PIDSettings, dt: float) -> PIDController:
    return PIDController(
        kp=pid_settings.kp,
        ki=pid_settings.ki,
        kd=pid_settings.kd,
        dt=dt,
        integral_limits=pid_settings.integral_limits,
    )


def _apply_target_schedule(
    env: PID1DEnv,
    step: int,
    target_offsets: Sequence[float] | None,
    segment_steps: int | None,
) -> None:
    if target_offsets is None:
        return
    if segment_steps is None or segment_steps < 1:
        raise ValueError("segment_steps must be >= 1 when target_offsets is set")
    index = min(step // segment_steps, len(target_offsets) - 1)
    target = env.config.target + float(target_offsets[index])
    if env.config.bounds is not None:
        low, high = env.config.bounds
        target = min(max(target, low), high)
    env.set_target(target)


def rollout(
    pid_settings: PIDSettings,
    env_config: EnvironmentConfig,
    hyperparameters: LossHyperparameters,
    *,
    seed: int = 0,
    record: bool = True,
    target_offsets: Sequence[float] | None = None,
    segment_steps: int | None = None,
) -> RolloutResult:
    """Run one closed-loop episode and optionally record its trajectory.

    ``target_offsets`` supplies a piecewise-constant offset added to
    ``env_config.target``.  The offset changes every ``segment_steps`` steps
    and is clamped to ``env_config.bounds`` when bounds are set.
    """
    if target_offsets is not None and (
        segment_steps is None or segment_steps < 1
    ):
        raise ValueError("segment_steps must be >= 1 when target_offsets is set")

    config = copy.deepcopy(env_config)
    env = PID1DEnv(config)
    env.reset(seed=seed)

    pid = _make_pid(pid_settings, config.dt)
    previous_velocity = env.velocity
    total_cost = 0.0
    steps_run = 0
    terminated = False
    truncated = False

    times: list[float] = []
    positions: list[float] = []
    velocities: list[float] = []
    targets: list[float] = []
    errors: list[float] = []
    actions: list[float] = []

    while steps_run < config.max_steps:
        _apply_target_schedule(env, steps_run, target_offsets, segment_steps)

        action = pid(env.error)
        _, _, terminated, truncated, info = env.step(action)

        acceleration = (info["velocity"] - previous_velocity) / config.dt
        previous_velocity = info["velocity"]

        total_cost += (
            hyperparameters.position_error_weight * info["error"] ** 2
            + hyperparameters.velocity_weight * info["velocity"] ** 2
            + hyperparameters.acceleration_weight * acceleration**2
        )
        steps_run += 1

        if record:
            times.append(info["time"])
            positions.append(info["position"])
            velocities.append(info["velocity"])
            targets.append(info["target"])
            errors.append(info["error"])
            actions.append(action)

        if truncated:
            break
        if terminated:
            remaining = config.max_steps - steps_run
            total_cost += remaining * (
                hyperparameters.position_error_weight * info["error"] ** 2
                + hyperparameters.velocity_weight * info["velocity"] ** 2
                + hyperparameters.acceleration_weight * acceleration**2
            )
            steps_run = config.max_steps
            break

    trajectory = None
    if record:
        trajectory = Trajectory(
            time=np.asarray(times, dtype=np.float32),
            position=np.asarray(positions, dtype=np.float32),
            velocity=np.asarray(velocities, dtype=np.float32),
            target=np.asarray(targets, dtype=np.float32),
            error=np.asarray(errors, dtype=np.float32),
            action=np.asarray(actions, dtype=np.float32),
            dt=float(config.dt),
        )

    return RolloutResult(
        loss=total_cost / config.max_steps,
        terminated=terminated,
        truncated=truncated,
        steps=steps_run,
        seed=seed,
        trajectory=trajectory,
    )


def evaluate(
    pid_settings: PIDSettings,
    env_config: EnvironmentConfig,
    hyperparameters: LossHyperparameters,
    *,
    seeds: Sequence[int] = (0,),
    record: bool = False,
) -> float:
    """Return mean per-step loss across ``seeds``."""
    if len(seeds) == 0:
        raise ValueError("seeds must contain at least one value")
    losses = [
        rollout(
            pid_settings,
            env_config,
            hyperparameters,
            seed=int(seed),
            record=record,
        ).loss
        for seed in seeds
    ]
    return float(np.mean(losses))


def pid_loss(
    pid_settings: PIDSettings,
    env_config: EnvironmentConfig,
    hyperparameters: LossHyperparameters,
    *,
    seed: int = 0,
    episodes: int = 1,
) -> float:
    """Backwards-compatible loss wrapper around :func:`evaluate`."""
    if episodes < 1:
        raise ValueError("episodes must be >= 1")
    return evaluate(
        pid_settings,
        env_config,
        hyperparameters,
        seeds=range(seed, seed + episodes),
    )
