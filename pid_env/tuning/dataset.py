"""Sharded dataset generation for the future gains-prediction transformer.

Each configuration is labelled once with CMA-ES.  The input samples are
closed-loop probe trajectories collected under *freshly randomised* prior
gains, so the model learns to improve from arbitrary current controllers.
"""

from __future__ import annotations

import concurrent.futures
import json
import math
import multiprocessing
import subprocess
import sys
from dataclasses import replace
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Callable

import numpy as np

from ..config import EnvironmentConfig
from ..evaluation import LossHyperparameters, TRAJECTORY_FEATURES
from .cmaes import tune
from .probe import sample_probe
from .sampler import (
    ENV_FEATURE_NAMES,
    env_feature_vector,
    sample_environment,
)
from .serialization import (
    cmaes_config_to_dict,
    dataset_config_to_dict,
    env_sampler_to_dict,
    gain_sampler_to_dict,
    load_json,
    loss_hyperparameters_to_dict,
    pid_settings_to_dict,
    probe_config_to_dict,
    save_json,
)
from .types import (
    CMAESConfig,
    DatasetConfig,
    EnvSamplerConfig,
    GainSamplerConfig,
    ProbeConfig,
)

SCHEMA_VERSION = 1


def _split_for_index(config: DatasetConfig, index: int) -> str:
    rng = np.random.default_rng(config.seed + 104_729 * (index + 1))
    value = rng.random()
    if value < config.train_fraction:
        return "train"
    if value < config.train_fraction + config.validation_fraction:
        return "val"
    return "test"


def _pad_features(features: np.ndarray, horizon: int) -> tuple[np.ndarray, int]:
    length = min(int(features.shape[0]), horizon)
    padded = np.zeros((horizon, features.shape[1]), dtype=np.float32)
    padded[:length] = features[:length]
    return padded, length


def _git_commit() -> str | None:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=Path(__file__).resolve().parents[2],
                stderr=subprocess.DEVNULL,
                text=True,
            )
            .strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return None


def _versions() -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for name in ("numpy", "cma"):
        try:
            result[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            result[name] = None
    return result


def _env_config_json(config: EnvironmentConfig) -> str:
    from .serialization import environment_config_to_dict

    return json.dumps(
        environment_config_to_dict(config), sort_keys=True, separators=(",", ":")
    )


def build_shard(
    shard_index: int,
    dataset_config: DatasetConfig,
    sampling_config: EnvSamplerConfig,
    gain_sampler: GainSamplerConfig,
    probe_config: ProbeConfig,
    cma_config: CMAESConfig,
    hyperparameters: LossHyperparameters,
) -> dict[str, Any]:
    """Generate one shard of samples.  Safe to run in a worker process."""
    start = shard_index * dataset_config.shard_size
    stop = min(start + dataset_config.shard_size, dataset_config.num_configs)
    output_dir = Path(dataset_config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    shard_path = output_dir / f"shard_{shard_index:04d}.npz"

    trajectory_rows: list[np.ndarray] = []
    length_rows: list[int] = []
    prior_gain_rows: list[np.ndarray] = []
    env_feature_rows: list[np.ndarray] = []
    best_gain_rows: list[np.ndarray] = []
    best_loss_rows: list[float] = []
    validation_loss_rows: list[float] = []
    prior_loss_rows: list[float] = []
    terminated_rows: list[bool] = []
    config_id_rows: list[int] = []
    probe_seed_rows: list[int] = []
    prior_index_rows: list[int] = []
    split_rows: list[str] = []
    config_json_rows: list[str] = []

    for config_index in range(start, stop):
        rng = np.random.default_rng(dataset_config.seed + config_index)
        env_config = sample_environment(
            rng,
            sampling_config,
            horizon_steps=dataset_config.horizon_steps,
        )
        split = _split_for_index(dataset_config, config_index)
        config_json = _env_config_json(env_config)

        per_config_cma = replace(
            cma_config,
            seed=cma_config.seed + config_index,
            workers=1,
        )
        tuning_result = tune(env_config, hyperparameters, per_config_cma)
        best_gains = tuning_result.best_gains
        best_gain_vector = np.asarray(
            [best_gains.kp, best_gains.ki, best_gains.kd], dtype=np.float32
        )
        env_features = env_feature_vector(env_config)

        for probe_seed in probe_config.probe_seeds:
            for prior_index in range(probe_config.priors_per_config):
                result, prior_gains = sample_probe(
                    env_config,
                    hyperparameters,
                    probe_config,
                    gain_sampler,
                    rng,
                    seed=probe_seed,
                )
                assert result.trajectory is not None
                features = result.trajectory.features()
                padded, length = _pad_features(features, dataset_config.horizon_steps)

                trajectory_rows.append(padded)
                length_rows.append(length)
                prior_gain_rows.append(
                    np.asarray(
                        [prior_gains.kp, prior_gains.ki, prior_gains.kd],
                        dtype=np.float32,
                    )
                )
                env_feature_rows.append(env_features)
                best_gain_rows.append(best_gain_vector)
                best_loss_rows.append(float(tuning_result.best_loss))
                validation_loss_rows.append(float(tuning_result.validation_loss))
                prior_loss_rows.append(float(result.loss))
                terminated_rows.append(bool(result.terminated))
                config_id_rows.append(config_index)
                probe_seed_rows.append(int(probe_seed))
                prior_index_rows.append(prior_index)
                split_rows.append(split)
                config_json_rows.append(config_json)

    if not trajectory_rows:
        raise RuntimeError("no samples were generated for shard")

    payload = {
        "probe_trajectory": np.stack(trajectory_rows).astype(np.float32),
        "lengths": np.asarray(length_rows, dtype=np.int32),
        "prior_gains": np.stack(prior_gain_rows).astype(np.float32),
        "env_features": np.stack(env_feature_rows).astype(np.float32),
        "best_gains": np.stack(best_gain_rows).astype(np.float32),
        "best_loss": np.asarray(best_loss_rows, dtype=np.float32),
        "validation_loss": np.asarray(validation_loss_rows, dtype=np.float32),
        "prior_loss": np.asarray(prior_loss_rows, dtype=np.float32),
        "terminated": np.asarray(terminated_rows, dtype=bool),
        "config_id": np.asarray(config_id_rows, dtype=np.int64),
        "probe_seed": np.asarray(probe_seed_rows, dtype=np.int64),
        "prior_index": np.asarray(prior_index_rows, dtype=np.int16),
        "split": np.asarray(split_rows, dtype="U5"),
        "config_json": np.asarray(config_json_rows, dtype="U4096"),
    }
    np.savez(shard_path, **payload)
    return {
        "shard_index": shard_index,
        "path": str(shard_path),
        "configs": stop - start,
        "samples": len(trajectory_rows),
    }


def _dataset_signature(
    config: DatasetConfig,
    sampling: EnvSamplerConfig,
    gain_sampler: GainSamplerConfig,
    probe: ProbeConfig,
    cma: CMAESConfig,
    hyperparameters: LossHyperparameters,
) -> dict[str, Any]:
    dataset_values = dataset_config_to_dict(config)
    dataset_values.pop("resume", None)
    dataset_values.pop("workers", None)
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset_config": dataset_values,
        "sampler_config": env_sampler_to_dict(sampling),
        "gain_sampler_config": gain_sampler_to_dict(gain_sampler),
        "probe_config": probe_config_to_dict(probe),
        "cma_config": cmaes_config_to_dict(cma),
        "loss_hyperparameters": loss_hyperparameters_to_dict(hyperparameters),
    }


def _compute_stats(shard_paths: list[Path]) -> dict[str, Any]:
    env_sum = np.zeros(len(ENV_FEATURE_NAMES), dtype=np.float64)
    env_sumsq = np.zeros(len(ENV_FEATURE_NAMES), dtype=np.float64)
    env_count = 0

    prior_sum = np.zeros(3, dtype=np.float64)
    prior_sumsq = np.zeros(3, dtype=np.float64)
    best_sum = np.zeros(3, dtype=np.float64)
    best_sumsq = np.zeros(3, dtype=np.float64)
    gain_count = 0

    traj_sum = np.zeros(len(TRAJECTORY_FEATURES), dtype=np.float64)
    traj_sumsq = np.zeros(len(TRAJECTORY_FEATURES), dtype=np.float64)
    traj_count = 0

    split_counts: dict[str, int] = {"train": 0, "val": 0, "test": 0}
    total_samples = 0

    for path in shard_paths:
        with np.load(path, allow_pickle=False) as data:
            env = data["env_features"].astype(np.float64)
            env_sum += env.sum(axis=0)
            env_sumsq += (env**2).sum(axis=0)
            env_count += env.shape[0]

            prior = data["prior_gains"].astype(np.float64)
            prior_sum += prior.sum(axis=0)
            prior_sumsq += (prior**2).sum(axis=0)
            best = data["best_gains"].astype(np.float64)
            best_sum += best.sum(axis=0)
            best_sumsq += (best**2).sum(axis=0)
            gain_count += best.shape[0]

            traj = data["probe_trajectory"].astype(np.float64)
            lengths = data["lengths"].astype(np.int64)
            traj_sum += traj.sum(axis=(0, 1))
            traj_sumsq += (traj**2).sum(axis=(0, 1))
            traj_count += int(lengths.sum())

            for split, count in zip(*np.unique(data["split"], return_counts=True)):
                split_counts[str(split)] = split_counts.get(str(split), 0) + int(count)
            total_samples += int(data["split"].size)

    def summarize(
        total: np.ndarray, total_sq: np.ndarray, count: int, names: tuple[str, ...]
    ) -> dict[str, dict[str, list[float]]]:
        if count == 0:
            mean = np.zeros_like(total)
            std = np.ones_like(total)
        else:
            mean = total / count
            variance = np.maximum(total_sq / count - mean**2, 0.0)
            std = np.sqrt(variance)
            std[std < 1e-8] = 1.0
        return {
            "mean": {name: float(value) for name, value in zip(names, mean)},
            "std": {name: float(value) for name, value in zip(names, std)},
        }

    return {
        "samples": total_samples,
        "split_counts": split_counts,
        "env_features": summarize(
            env_sum, env_sumsq, env_count, ENV_FEATURE_NAMES
        ),
        "prior_gains": summarize(
            prior_sum, prior_sumsq, gain_count, ("kp", "ki", "kd")
        ),
        "best_gains": summarize(
            best_sum, best_sumsq, gain_count, ("kp", "ki", "kd")
        ),
        "trajectory_features": summarize(
            traj_sum, traj_sumsq, traj_count, TRAJECTORY_FEATURES
        ),
    }


def _finalize(
    config: DatasetConfig,
    signature: dict[str, Any],
    shard_results: list[dict[str, Any]],
) -> dict[str, Any]:
    output_dir = Path(config.output_dir)
    shard_paths = sorted(output_dir.glob("shard_*.npz"))
    if not shard_paths:
        raise RuntimeError("no shards found to finalize")

    manifest = {
        **signature,
        "name": config.name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "package_versions": _versions(),
        "python_version": sys.version,
        "feature_names": list(TRAJECTORY_FEATURES),
        "env_feature_names": list(ENV_FEATURE_NAMES),
        "num_shards": len(shard_paths),
        "shards": shard_results,
        "stats": _compute_stats(shard_paths),
        "complete": True,
    }
    save_json(output_dir / "manifest.json", manifest)
    return manifest


def generate_dataset(
    config: DatasetConfig,
    hyperparameters: LossHyperparameters | None = None,
    *,
    sampling_config: EnvSamplerConfig | None = None,
    gain_sampler: GainSamplerConfig | None = None,
    probe_config: ProbeConfig | None = None,
    cma_config: CMAESConfig | None = None,
    callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Generate all shards and write ``manifest.json``.

    With ``config.resume=True``, existing shard files are reused.  The
    manifest is written first in an incomplete state so interrupted runs are
    easy to inspect.
    """
    hyperparameters = hyperparameters or LossHyperparameters()
    sampling_config = sampling_config or EnvSamplerConfig()
    gain_sampler = gain_sampler or GainSamplerConfig()
    probe_config = probe_config or ProbeConfig()
    cma_config = cma_config or CMAESConfig()

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    if not config.resume:
        for stale in output_dir.glob("shard_*.npz"):
            stale.unlink()

    signature = _dataset_signature(
        config,
        sampling_config,
        gain_sampler,
        probe_config,
        cma_config,
        hyperparameters,
    )

    if manifest_path.exists() and config.resume:
        previous = load_json(manifest_path)
        previous_signature = {
            key: previous.get(key) for key in signature
        }
        if previous_signature != signature:
            raise ValueError(
                "existing dataset manifest does not match the requested "
                "configuration; use a new output_dir or resume=False"
            )

    save_json(
        manifest_path,
        {
            **signature,
            "name": config.name,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "git_commit": _git_commit(),
            "package_versions": _versions(),
            "python_version": sys.version,
            "feature_names": list(TRAJECTORY_FEATURES),
            "env_feature_names": list(ENV_FEATURE_NAMES),
            "complete": False,
        },
    )

    num_shards = math.ceil(config.num_configs / config.shard_size)
    pending: list[int] = []
    for shard_index in range(num_shards):
        path = output_dir / f"shard_{shard_index:04d}.npz"
        if config.resume and path.exists():
            continue
        pending.append(shard_index)

    payloads = [
        (
            shard_index,
            config,
            sampling_config,
            gain_sampler,
            probe_config,
            cma_config,
            hyperparameters,
        )
        for shard_index in pending
    ]

    shard_results: list[dict[str, Any]] = []

    if config.workers == 1:
        for payload in payloads:
            result = build_shard(*payload)
            shard_results.append(result)
            if callback is not None:
                callback(result)
    else:
        try:
            mp_context = multiprocessing.get_context("fork")
        except ValueError:
            mp_context = None
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=config.workers,
            mp_context=mp_context,
        ) as executor:
            futures = {
                executor.submit(build_shard, *payload): payload[0]
                for payload in payloads
            }
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                shard_results.append(result)
                if callback is not None:
                    callback(result)

    if config.resume:
        for shard_index in range(num_shards):
            path = output_dir / f"shard_{shard_index:04d}.npz"
            if not path.exists():
                raise RuntimeError(f"missing shard after generation: {path}")
            if not any(item["shard_index"] == shard_index for item in shard_results):
                with np.load(path, allow_pickle=False) as data:
                    samples = int(data["split"].size)
                    configs = int(np.unique(data["config_id"]).size)
                shard_results.append(
                    {
                        "shard_index": shard_index,
                        "path": str(path),
                        "reused": True,
                        "samples": samples,
                        "configs": configs,
                    }
                )

    shard_results.sort(key=lambda item: item["shard_index"])
    return _finalize(config, signature, shard_results)
