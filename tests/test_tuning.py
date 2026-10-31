from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

from pid_env import (
    EnvironmentConfig,
    LossHyperparameters,
    PIDSettings,
    evaluate,
    pid_loss,
    rollout,
)
from pid_env.tuning import (
    CMAESConfig,
    DatasetConfig,
    EnvSamplerConfig,
    GainSamplerConfig,
    ProbeConfig,
    decode_gains,
    encode_gains,
    generate_dataset,
    probe_rollout,
    sample_environment,
    sample_prior_gains,
    tune,
)
from pid_env.tuning.sampler import env_feature_vector
from pid_env.tuning.serialization import (
    cmaes_config_from_dict,
    cmaes_config_to_dict,
    dataset_config_from_dict,
    dataset_config_to_dict,
    env_sampler_from_dict,
    env_sampler_to_dict,
    environment_config_from_dict,
    environment_config_to_dict,
    gain_sampler_from_dict,
    gain_sampler_to_dict,
    loss_hyperparameters_from_dict,
    loss_hyperparameters_to_dict,
    pid_settings_from_dict,
    pid_settings_to_dict,
    probe_config_from_dict,
    probe_config_to_dict,
)

HAS_CMA = importlib.util.find_spec("cma") is not None


class SerializationTests(unittest.TestCase):
    def test_environment_round_trip(self) -> None:
        config = EnvironmentConfig(
            dt=0.05,
            max_steps=321,
            mass=1.7,
            weight=12.3,
            drag=0.4,
            thrust=30.0,
            target=1.4,
            disturbance_strength=2.0,
            target_noise_strength=0.3,
            bounds=(-2.0, 4.0),
        )
        self.assertEqual(
            environment_config_from_dict(environment_config_to_dict(config)),
            config,
        )

    def test_pid_and_loss_round_trip(self) -> None:
        settings = PIDSettings(1.0, 0.2, 0.3, integral_limits=(-1.0, 1.0))
        self.assertEqual(
            pid_settings_from_dict(pid_settings_to_dict(settings)), settings
        )
        hyper = LossHyperparameters(1.0, 0.2, 0.05)
        self.assertEqual(
            loss_hyperparameters_from_dict(loss_hyperparameters_to_dict(hyper)),
            hyper,
        )

    def test_tuning_configs_round_trip(self) -> None:
        self.assertEqual(
            cmaes_config_from_dict(
                cmaes_config_to_dict(CMAESConfig(max_evaluations=12))
            ).max_evaluations,
            12,
        )
        self.assertEqual(
            dataset_config_from_dict(
                dataset_config_to_dict(DatasetConfig(num_configs=7))
            ).num_configs,
            7,
        )
        self.assertEqual(
            env_sampler_from_dict(env_sampler_to_dict(EnvSamplerConfig())),
            EnvSamplerConfig(),
        )
        self.assertEqual(
            gain_sampler_from_dict(gain_sampler_to_dict(GainSamplerConfig())),
            GainSamplerConfig(),
        )
        self.assertEqual(
            probe_config_from_dict(probe_config_to_dict(ProbeConfig())),
            ProbeConfig(),
        )


class RolloutTests(unittest.TestCase):
    def test_trajectory_recorded(self) -> None:
        config = EnvironmentConfig(max_steps=50)
        result = rollout(
            PIDSettings(1.0),
            config,
            LossHyperparameters(),
            seed=0,
            record=True,
        )
        self.assertIsNotNone(result.trajectory)
        assert result.trajectory is not None
        self.assertEqual(len(result.trajectory), 50)
        self.assertEqual(result.trajectory.features().shape, (50, 6))

    def test_matches_pid_loss(self) -> None:
        config = EnvironmentConfig(max_steps=40)
        hyper = LossHyperparameters(1.0, 0.1, 0.01)
        gains = PIDSettings(2.0, 0.3, 0.2)
        expected = pid_loss(gains, config, hyper, seed=3, episodes=2)
        actual = evaluate(gains, config, hyper, seeds=(3, 4))
        self.assertAlmostEqual(expected, actual)

    def test_target_schedule(self) -> None:
        config = EnvironmentConfig(
            max_steps=30,
            weight=0.0,
            drag=0.0,
            thrust=0.0,
            target=0.0,
        )
        result = rollout(
            PIDSettings(0.0),
            config,
            LossHyperparameters(),
            seed=0,
            record=True,
            target_offsets=(0.0, 1.0),
            segment_steps=10,
        )
        assert result.trajectory is not None
        targets = result.trajectory.target
        self.assertAlmostEqual(float(targets[0]), 0.0)
        self.assertAlmostEqual(float(targets[10]), 1.0)


class SamplerTests(unittest.TestCase):
    def test_environment_is_feasible(self) -> None:
        rng = np.random.default_rng(0)
        config = EnvSamplerConfig()
        for _ in range(20):
            env = sample_environment(rng, config, horizon_steps=123)
            self.assertEqual(env.max_steps, 123)
            self.assertGreater(env.thrust, env.weight)
            self.assertIsNotNone(env.bounds)
            assert env.bounds is not None
            self.assertLess(env.bounds[0], env.target)
            self.assertGreater(env.bounds[1], env.target)
            self.assertEqual(env_feature_vector(env).shape[0], 23)

    def test_prior_gains_ranges(self) -> None:
        rng = np.random.default_rng(0)
        config = GainSamplerConfig(zero_ki_probability=0.0, zero_kd_probability=0.0)
        for _ in range(20):
            gains = sample_prior_gains(rng, config)
            self.assertGreaterEqual(gains.kp, config.kp_range[0])
            self.assertLessEqual(gains.kp, config.kp_range[1])
            self.assertGreaterEqual(gains.ki, config.ki_range[0])
            self.assertGreaterEqual(gains.kd, config.kd_range[0])


@unittest.skipUnless(HAS_CMA, "cma not installed")
class CMAESTests(unittest.TestCase):
    def test_encoding_round_trip(self) -> None:
        config = CMAESConfig(gain_bounds=(1e-6, 1e3))
        settings = PIDSettings(1.2, 0.04, 3.0)
        decoded = decode_gains(encode_gains(settings, config), config)
        np.testing.assert_allclose(
            [decoded.kp, decoded.ki, decoded.kd],
            [settings.kp, settings.ki, settings.kd],
            rtol=1e-9,
        )

    def test_tuning_improves_and_respects_bounds(self) -> None:
        env = EnvironmentConfig(max_steps=100, disturbance_strength=0.0)
        hyper = LossHyperparameters(1.0, 0.1, 0.01)
        start = PIDSettings(0.01, 0.0, 0.0)
        initial_loss = pid_loss(start, env, hyper, seed=0, episodes=2)
        config = CMAESConfig(
            initial_gains=start,
            max_evaluations=50,
            seeds=(0, 1),
            validation_seeds=(100, 101),
            popsize=8,
            sigma0=1.0,
            seed=0,
        )
        result = tune(env, hyper, config)
        self.assertLess(result.best_loss, initial_loss)
        self.assertGreaterEqual(result.best_gains.kp, config.gain_bounds[0])
        self.assertLessEqual(result.best_gains.kp, config.gain_bounds[1])
        self.assertGreaterEqual(result.best_gains.ki, config.gain_bounds[0])
        self.assertGreaterEqual(result.best_gains.kd, config.gain_bounds[0])

    def test_infeasible_environment_rejected(self) -> None:
        env = EnvironmentConfig(weight=20.0, thrust=10.0)
        with self.assertRaises(ValueError):
            tune(env, LossHyperparameters(), CMAESConfig(max_evaluations=5))


@unittest.skipUnless(HAS_CMA, "cma not installed")
class DatasetTests(unittest.TestCase):
    def test_probe_and_dataset_shapes(self) -> None:
        rng = np.random.default_rng(0)
        env = sample_environment(rng, EnvSamplerConfig(), horizon_steps=40)
        hyper = LossHyperparameters()
        probe = ProbeConfig(
            target_offsets=(0.0, 0.5),
            segment_steps=10,
            priors_per_config=1,
            probe_seeds=(0,),
            max_prior_attempts=2,
        )
        gains = PIDSettings(1.0, 0.1, 0.1)
        result = probe_rollout(env, gains, hyper, probe, seed=0)
        self.assertIsNotNone(result.trajectory)
        assert result.trajectory is not None
        self.assertEqual(result.trajectory.features().shape, (40, 6))

        with tempfile.TemporaryDirectory() as temp_dir:
            config = DatasetConfig(
                output_dir=temp_dir,
                num_configs=2,
                shard_size=1,
                horizon_steps=40,
                workers=1,
                resume=False,
            )
            cma = CMAESConfig(
                max_evaluations=8,
                seeds=(0,),
                validation_seeds=(1,),
                popsize=5,
                sigma0=1.0,
            )
            manifest = generate_dataset(
                config,
                hyper,
                probe_config=probe,
                cma_config=cma,
            )
            self.assertTrue(manifest["complete"])
            self.assertEqual(manifest["stats"]["samples"], 2)
            self.assertEqual(manifest["num_shards"], 2)

            shard = Path(temp_dir) / "shard_0000.npz"
            self.assertTrue(shard.exists())
            with np.load(shard, allow_pickle=False) as data:
                self.assertEqual(
                    data["probe_trajectory"].shape, (1, 40, 6)
                )
                self.assertEqual(data["prior_gains"].shape, (1, 3))
                self.assertEqual(data["best_gains"].shape, (1, 3))
                self.assertEqual(data["env_features"].shape, (1, 23))
                self.assertTrue(np.isfinite(data["best_gains"]).all())


if __name__ == "__main__":
    unittest.main()
