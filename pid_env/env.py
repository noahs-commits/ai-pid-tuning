"""A small 1D physics environment for experimenting with PID tuning."""

from __future__ import annotations

from typing import Any

import numpy as np

from .config import EnvironmentConfig
from .noise import FractalPerlinNoise

Observation = np.ndarray


class PID1DEnv:
    """One-dimensional mass-spring-free body with thrust, drag and noise.

    The agent controls a vertical thruster with a scalar ``action`` in
    ``[-1, 1]``.  The realised thrust is ``action * config.thrust``.  Gravity
    (``config.weight``), linear drag and a fractal-Perlin noise disturbance
    act on the body.

    The observation is ``[position, velocity, target, error]`` where
    ``error = target - position``.

    ``step`` follows the Gymnasium convention::

        obs, reward, terminated, truncated, info = env.step(action)
    """

    def __init__(self, config: EnvironmentConfig | None = None) -> None:
        self.config = config or EnvironmentConfig()
        self._build_noise(
            self.config.disturbance_noise.seed,
            self.config.target_noise.seed,
        )
        self._rng = np.random.default_rng(self.config.disturbance_noise.seed)

        self.position = self.config.initial_position
        self.velocity = self.config.initial_velocity
        self.target = self.config.target
        self.time = 0.0
        self.steps = 0
        self._episode_seed = self.config.disturbance_noise.seed

    # ------------------------------------------------------------------
    # Noise / helpers
    # ------------------------------------------------------------------
    def _build_noise(self, disturbance_seed: int, target_seed: int) -> None:
        d = self.config.disturbance_noise
        t = self.config.target_noise
        self._disturbance_noise = FractalPerlinNoise(
            seed=disturbance_seed,
            octaves=d.octaves,
            persistence=d.persistence,
            lacunarity=d.lacunarity,
            frequency=d.frequency,
        )
        self._target_noise = FractalPerlinNoise(
            seed=target_seed,
            octaves=t.octaves,
            persistence=t.persistence,
            lacunarity=t.lacunarity,
            frequency=t.frequency,
        )

    @property
    def error(self) -> float:
        return self.target - self.position

    @property
    def observation(self) -> Observation:
        return np.array(
            [self.position, self.velocity, self.target, self.error],
            dtype=np.float64,
        )

    def _current_disturbance(self) -> float:
        if self.config.disturbance_strength == 0.0:
            return 0.0
        n = self._disturbance_noise.sample(
            self.time * self.config.disturbance_frequency
        )
        return self.config.disturbance_strength * float(n)

    def _current_target(self) -> float:
        if self.config.target_noise_strength == 0.0:
            return self.config.target
        n = self._target_noise.sample(self.time * self.config.target_noise_frequency)
        return self.config.target + self.config.target_noise_strength * float(n)

    def _info(self, disturbance: float, thrust_force: float, drag_force: float) -> dict[str, Any]:
        return {
            "time": self.time,
            "steps": self.steps,
            "position": self.position,
            "velocity": self.velocity,
            "target": self.target,
            "error": self.error,
            "thrust_force": thrust_force,
            "drag_force": drag_force,
            "disturbance_force": disturbance,
        }

    # ------------------------------------------------------------------
    # Gym-like API
    # ------------------------------------------------------------------
    def reset(
        self,
        *,
        seed: int | None = None,
        randomize: bool = False,
    ) -> tuple[Observation, dict[str, Any]]:
        """Reset the episode.

        Parameters
        ----------
        seed:
            Optional new noise seed.  Results are reproducible for a given
            seed and action sequence.
        randomize:
            Randomise the initial position/velocity slightly around their
            configured values.  Uses ``seed`` when provided.
        """
        if seed is not None:
            self._episode_seed = int(seed)
            self._rng = np.random.default_rng(self._episode_seed)
            self._build_noise(self._episode_seed, self._episode_seed + 1_000_003)

        self.time = 0.0
        self.steps = 0
        self.target = self.config.target
        self.position = self.config.initial_position
        self.velocity = self.config.initial_velocity

        if randomize:
            self.position += float(self._rng.uniform(-0.5, 0.5))
            self.velocity += float(self._rng.uniform(-0.5, 0.5))

        info = self._info(0.0, 0.0, 0.0)
        return self.observation, info

    def step(
        self,
        action: float | np.ndarray,
    ) -> tuple[Observation, float, bool, bool, dict[str, Any]]:
        """Advance the simulation by ``config.dt`` seconds.

        ``action`` is clipped to ``[-1, 1]`` then multiplied by
        ``config.thrust``.
        """
        if isinstance(action, np.ndarray):
            action = float(action.item())
        action = float(np.clip(action, -1.0, 1.0))

        disturbance = self._current_disturbance()
        self.target = self._current_target()

        thrust_force = action * self.config.thrust
        drag_force = -self.config.drag * self.velocity
        net_force = thrust_force - self.config.weight + drag_force + disturbance
        acceleration = net_force / self.config.mass

        self.velocity += acceleration * self.config.dt
        self.position += self.velocity * self.config.dt
        self.time += self.config.dt
        self.steps += 1

        error = self.error
        reward = -(error**2) - self.config.reward_control_cost * action**2

        terminated = False
        if self.config.bounds is not None:
            low, high = self.config.bounds
            terminated = not (low <= self.position <= high)
        truncated = self.steps >= self.config.max_steps

        info = self._info(disturbance, thrust_force, drag_force)
        return self.observation, float(reward), terminated, truncated, info

    def set_target(self, target: float) -> None:
        """Set a new static target (until target noise is enabled)."""
        self.config.target = float(target)
        self.target = float(target)
