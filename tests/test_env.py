from __future__ import annotations

import copy
import importlib.util
import unittest

import numpy as np

from pid_env import (
    EnvironmentConfig,
    FractalPerlinNoise,
    LossHyperparameters,
    PID1DEnv,
    PIDController,
    PIDSettings,
    pid_loss,
    plot_trajectory,
)

HAS_MATPLOTLIB = importlib.util.find_spec("matplotlib") is not None


class NoiseTests(unittest.TestCase):
    def test_deterministic(self) -> None:
        a = FractalPerlinNoise(seed=42)
        b = FractalPerlinNoise(seed=42)
        xs = np.linspace(0.0, 10.0, 100)
        np.testing.assert_allclose(a.sample(xs), b.sample(xs))

    def test_different_seeds_differ(self) -> None:
        xs = np.linspace(0.0, 10.0, 100)
        a = FractalPerlinNoise(seed=1).sample(xs)
        b = FractalPerlinNoise(seed=2).sample(xs)
        self.assertFalse(np.allclose(a, b))

    def test_bounded(self) -> None:
        noise = FractalPerlinNoise(seed=7, octaves=6)
        xs = np.linspace(-1000.0, 1000.0, 5000)
        samples = noise.sample(xs)
        self.assertGreaterEqual(float(samples.min()), -1.0)
        self.assertLessEqual(float(samples.max()), 1.0)

    def test_smooth(self) -> None:
        noise = FractalPerlinNoise(seed=3)
        xs = np.linspace(0.0, 20.0, 2000)
        diffs = np.abs(np.diff(noise.sample(xs)))
        self.assertLess(float(diffs.max()), 0.1)


class PhysicsTests(unittest.TestCase):
    def test_thrust_accelerates_up(self) -> None:
        cfg = EnvironmentConfig(weight=0.0, drag=0.0, thrust=10.0, mass=1.0)
        env = PID1DEnv(cfg)
        env.reset(seed=0)
        _, _, _, _, info = env.step(1.0)
        self.assertGreater(info["velocity"], 0.0)
        self.assertGreater(info["position"], 0.0)

    def test_weight_pulls_down(self) -> None:
        cfg = EnvironmentConfig(weight=9.81, drag=0.0, thrust=0.0)
        env = PID1DEnv(cfg)
        env.reset(seed=0)
        _, _, _, _, info = env.step(0.0)
        self.assertLess(info["velocity"], 0.0)
        self.assertLess(info["position"], 0.0)

    def test_drag_opposes_motion(self) -> None:
        cfg = EnvironmentConfig(weight=0.0, drag=2.0, thrust=10.0)
        env = PID1DEnv(cfg)
        env.reset(seed=0)
        env.velocity = 1.0
        _, _, _, _, info = env.step(0.0)
        self.assertLess(info["drag_force"], 0.0)

    def test_action_clipped(self) -> None:
        cfg = EnvironmentConfig(weight=0.0, drag=0.0, thrust=10.0)
        env = PID1DEnv(cfg)
        env.reset(seed=0)
        _, _, _, _, info = env.step(100.0)
        self.assertAlmostEqual(info["thrust_force"], 10.0)

    def test_reward_negative(self) -> None:
        env = PID1DEnv(EnvironmentConfig(weight=0.0, drag=0.0, thrust=0.0, target=1.0))
        env.reset(seed=0)
        _, reward, _, _, _ = env.step(0.0)
        self.assertLess(reward, 0.0)


class EpisodeTests(unittest.TestCase):
    def test_deterministic_reset(self) -> None:
        cfg = EnvironmentConfig(disturbance_strength=5.0)
        env = PID1DEnv(cfg)
        env.reset(seed=123)
        first = [env.step(0.2)[0].tolist() for _ in range(20)]
        env.reset(seed=123)
        second = [env.step(0.2)[0].tolist() for _ in range(20)]
        self.assertEqual(first, second)

    def test_disturbance_changes_physics(self) -> None:
        cfg = EnvironmentConfig(
            weight=0.0, drag=0.0, thrust=0.0, disturbance_strength=10.0
        )
        env = PID1DEnv(cfg)
        env.reset(seed=0)
        _, _, _, _, info = env.step(0.0)
        self.assertNotEqual(info["disturbance_force"], 0.0)

    def test_target_noise_moves_setpoint(self) -> None:
        cfg = EnvironmentConfig(target=1.0, target_noise_strength=0.5)
        env = PID1DEnv(cfg)
        env.reset(seed=0)
        _, _, _, _, info = env.step(0.0)
        self.assertNotEqual(info["target"], 1.0)

    def test_truncates_at_max_steps(self) -> None:
        env = PID1DEnv(EnvironmentConfig(max_steps=3))
        env.reset(seed=0)
        for _ in range(2):
            _, _, terminated, truncated, _ = env.step(0.0)
            self.assertFalse(truncated)
        _, _, _, truncated, _ = env.step(0.0)
        self.assertTrue(truncated)

    def test_bounds_terminate(self) -> None:
        cfg = EnvironmentConfig(
            weight=100.0, drag=0.0, thrust=0.0, bounds=(-1.0, 1.0), max_steps=1000
        )
        env = PID1DEnv(cfg)
        env.reset(seed=0)
        terminated = False
        for _ in range(1000):
            _, _, terminated, truncated, _ = env.step(0.0)
            if terminated or truncated:
                break
        self.assertTrue(terminated)

    def test_set_target(self) -> None:
        env = PID1DEnv(EnvironmentConfig(target=1.0))
        env.reset(seed=0)
        env.set_target(2.5)
        self.assertEqual(env.target, 2.5)
        self.assertAlmostEqual(env.error, 2.5)


class LossTests(unittest.TestCase):
    def setUp(self) -> None:
        self.env_config = EnvironmentConfig(
            target=1.0,
            max_steps=400,
            disturbance_strength=2.0,
        )

    def test_zero_weights_gives_zero(self) -> None:
        loss = pid_loss(
            PIDSettings(kp=1.0),
            self.env_config,
            LossHyperparameters(0.0, 0.0, 0.0),
        )
        self.assertEqual(loss, 0.0)

    def test_better_gains_have_lower_loss(self) -> None:
        hyper = LossHyperparameters(
            position_error_weight=1.0,
            velocity_weight=0.1,
            acceleration_weight=0.01,
        )
        good = pid_loss(PIDSettings(kp=3.0, ki=0.8, kd=1.2), self.env_config, hyper)
        bad = pid_loss(PIDSettings(kp=0.0), self.env_config, hyper)
        self.assertLess(good, bad)

    def test_deterministic(self) -> None:
        hyper = LossHyperparameters(1.0, 0.1, 0.01)
        a = pid_loss(PIDSettings(kp=2.0, ki=0.5), self.env_config, hyper, seed=5)
        b = pid_loss(PIDSettings(kp=2.0, ki=0.5), self.env_config, hyper, seed=5)
        self.assertEqual(a, b)

    def test_does_not_mutate_environment_config(self) -> None:
        env_config = copy.deepcopy(self.env_config)
        pid_loss(
            PIDSettings(kp=2.0), env_config, LossHyperparameters(), seed=1
        )
        self.assertEqual(env_config, self.env_config)

    def test_termination_extended_to_horizon(self) -> None:
        cfg = EnvironmentConfig(
            weight=100.0,
            thrust=0.0,
            drag=0.0,
            bounds=(-0.5, 0.5),
            max_steps=100,
        )
        loss = pid_loss(PIDSettings(kp=0.0), cfg, LossHyperparameters())
        self.assertGreater(loss, 0.0)

    def test_invalid_weights(self) -> None:
        with self.assertRaises(ValueError):
            LossHyperparameters(position_error_weight=-1.0)
        with self.assertRaises(ValueError):
            LossHyperparameters(velocity_weight=-1.0)
        with self.assertRaises(ValueError):
            LossHyperparameters(acceleration_weight=-1.0)

    def test_invalid_episodes(self) -> None:
        with self.assertRaises(ValueError):
            pid_loss(
                PIDSettings(kp=1.0),
                self.env_config,
                LossHyperparameters(),
                episodes=0,
            )


@unittest.skipUnless(HAS_MATPLOTLIB, "matplotlib not installed")
class PlottingTests(unittest.TestCase):
    def setUp(self) -> None:
        import matplotlib

        matplotlib.use("Agg", force=True)

    def test_position_and_setpoint(self) -> None:
        fig, ax = plot_trajectory([0.0, 0.5, 1.0], [1.0, 1.0, 1.0], show=False)
        try:
            labels = {line.get_label() for line in ax.get_lines()}
            self.assertEqual(labels, {"position", "setpoint"})
        finally:
            import matplotlib.pyplot as plt

            plt.close(fig)

    def test_extra_panels(self) -> None:
        fig, axes = plot_trajectory(
            [0.0, 0.5, 1.0],
            [1.0, 1.0, 1.0],
            velocities=[0.0, 1.0, 0.0],
            actions=[1.0, 0.5, 0.0],
            show=False,
        )
        try:
            self.assertEqual(len(axes), 3)
        finally:
            import matplotlib.pyplot as plt

            plt.close(fig)


class ControllerTests(unittest.TestCase):
    def test_output_limits(self) -> None:
        pid = PIDController(kp=100.0, dt=0.1)
        self.assertEqual(pid(1.0), 1.0)
        self.assertEqual(pid(-10.0), -1.0)

    def test_integral_limits(self) -> None:
        pid = PIDController(
            kp=0.0, ki=1.0, dt=1.0, integral_limits=(-0.5, 0.5)
        )
        for _ in range(10):
            pid(1.0)
        self.assertAlmostEqual(pid._integral, 0.5)


if __name__ == "__main__":
    unittest.main()
