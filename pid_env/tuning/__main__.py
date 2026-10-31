"""Command line entry point for CMA-ES tuning and dataset generation."""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any

from ..evaluation import rollout
from ..plotting import plot_trajectory
from .cmaes import tune
from .dataset import generate_dataset
from .serialization import (
    cmaes_config_from_dict,
    dataset_config_from_dict,
    env_sampler_from_dict,
    environment_config_from_dict,
    gain_sampler_from_dict,
    load_json,
    loss_hyperparameters_from_dict,
    probe_config_from_dict,
    save_json,
)


def _load_payload(path: str | None) -> dict[str, Any]:
    if path is None:
        return {}
    payload = load_json(path)
    if not isinstance(payload, dict):
        raise ValueError("configuration file must contain a JSON object")
    return payload


def _print_result(result) -> None:
    print(f"best gains    : kp={result.best_gains.kp:.6g} "
          f"ki={result.best_gains.ki:.6g} kd={result.best_gains.kd:.6g}")
    print(f"train loss    : {result.best_loss:.6g}")
    print(f"validation    : {result.validation_loss:.6g}")
    print(f"evaluations   : {result.evaluations}")
    print(f"generations   : {result.generations}")


def _cmd_cmaes(args: argparse.Namespace) -> None:
    payload = _load_payload(args.config)
    env_config = environment_config_from_dict(payload.get("environment", {}))
    hyperparameters = loss_hyperparameters_from_dict(
        payload.get("hyperparameters", {})
    )
    cma_config = cmaes_config_from_dict(payload.get("cmaes", {}))

    overrides: dict[str, Any] = {}
    for name in ("max_evaluations", "popsize", "sigma0", "seed", "restarts", "workers"):
        value = getattr(args, name, None)
        if value is not None:
            overrides[name] = value
    if args.seeds is not None:
        overrides["seeds"] = tuple(args.seeds)
    if args.validation_seeds is not None:
        overrides["validation_seeds"] = tuple(args.validation_seeds)
    if overrides:
        cma_config = replace(cma_config, **overrides)

    def progress(item: dict) -> None:
        if not args.quiet:
            gains = item["best_gains"]
            print(
                f"gen {item['generation']:>4} "
                f"evals {item['evaluations']:>5} "
                f"best {item['best_loss']:.6g} "
                f"kp={gains['kp']:.4g} ki={gains['ki']:.4g} kd={gains['kd']:.4g}"
            )

    result = tune(
        env_config,
        hyperparameters,
        cma_config,
        callback=progress,
    )
    _print_result(result)

    if args.out:
        save_json(
            args.out,
            {
                **result.to_dict(),
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        print(f"saved result  : {args.out}")

    if args.plot:
        validation_seed = cma_config.validation_seeds[0]
        episode = rollout(
            result.best_gains,
            env_config,
            hyperparameters,
            seed=validation_seed,
            record=True,
        )
        assert episode.trajectory is not None
        trajectory = episode.trajectory
        plot_trajectory(
            trajectory.position,
            trajectory.target,
            times=trajectory.time,
            velocities=trajectory.velocity,
            actions=trajectory.action,
            title="CMA-ES tuned PID",
            save_path=args.plot,
            show=False,
        )
        print(f"saved plot    : {args.plot}")


def _cmd_dataset(args: argparse.Namespace) -> None:
    payload = _load_payload(args.config)
    dataset_config = dataset_config_from_dict(payload.get("dataset", {}))
    sampling_config = env_sampler_from_dict(payload.get("sampler", {}))
    gain_sampler = gain_sampler_from_dict(payload.get("gain_sampler", {}))
    probe_config = probe_config_from_dict(payload.get("probe", {}))
    cma_config = cmaes_config_from_dict(payload.get("cmaes", {}))
    hyperparameters = loss_hyperparameters_from_dict(
        payload.get("hyperparameters", {})
    )

    dataset_overrides: dict[str, Any] = {}
    for name in (
        "output_dir",
        "num_configs",
        "shard_size",
        "horizon_steps",
        "seed",
        "workers",
        "name",
    ):
        value = getattr(args, name, None)
        if value is not None:
            dataset_overrides[name] = value
    if args.no_resume:
        dataset_overrides["resume"] = False
    if dataset_overrides:
        dataset_config = replace(dataset_config, **dataset_overrides)

    cma_overrides: dict[str, Any] = {}
    if args.max_evaluations is not None:
        cma_overrides["max_evaluations"] = args.max_evaluations
    if args.seeds is not None:
        cma_overrides["seeds"] = tuple(args.seeds)
    if cma_overrides:
        cma_config = replace(cma_config, **cma_overrides)

    probe_overrides: dict[str, Any] = {}
    if args.priors is not None:
        probe_overrides["priors_per_config"] = args.priors
    if args.probe_seeds is not None:
        probe_overrides["probe_seeds"] = tuple(args.probe_seeds)
    if probe_overrides:
        probe_config = replace(probe_config, **probe_overrides)

    def progress(item: dict) -> None:
        print(
            f"shard {item['shard_index']:>4} "
            f"configs {item.get('configs', 'reused')} "
            f"samples {item.get('samples', '')}"
        )

    manifest = generate_dataset(
        dataset_config,
        hyperparameters,
        sampling_config=sampling_config,
        gain_sampler=gain_sampler,
        probe_config=probe_config,
        cma_config=cma_config,
        callback=progress,
    )
    stats = manifest["stats"]
    print(f"dataset       : {dataset_config.output_dir}")
    print(f"samples       : {stats['samples']}")
    print(f"split counts  : {stats['split_counts']}")
    print(f"shards        : {manifest['num_shards']}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    cmaes = subparsers.add_parser("cmaes", help="tune PID gains for one config")
    cmaes.add_argument("--config", help="JSON file with environment/hyperparameters/cmaes")
    cmaes.add_argument("--out", help="write result JSON")
    cmaes.add_argument("--plot", help="save trajectory plot")
    cmaes.add_argument("--max-evals", dest="max_evaluations", type=int)
    cmaes.add_argument("--popsize", type=int)
    cmaes.add_argument("--sigma0", type=float)
    cmaes.add_argument("--seed", type=int)
    cmaes.add_argument("--restarts", type=int)
    cmaes.add_argument("--workers", type=int)
    cmaes.add_argument("--seeds", nargs="+", type=int)
    cmaes.add_argument("--validation-seeds", dest="validation_seeds", nargs="+", type=int)
    cmaes.add_argument("--quiet", action="store_true")
    cmaes.set_defaults(func=_cmd_cmaes)

    dataset = subparsers.add_parser("dataset", help="generate transformer dataset")
    dataset.add_argument("--config", help="JSON file with dataset config sections")
    dataset.add_argument("--out", dest="output_dir", help="output directory")
    dataset.add_argument("--name")
    dataset.add_argument("--num-configs", dest="num_configs", type=int)
    dataset.add_argument("--shard-size", dest="shard_size", type=int)
    dataset.add_argument("--horizon", dest="horizon_steps", type=int)
    dataset.add_argument("--seed", type=int)
    dataset.add_argument("--workers", type=int)
    dataset.add_argument("--max-evals", dest="max_evaluations", type=int)
    dataset.add_argument("--seeds", nargs="+", type=int)
    dataset.add_argument("--priors", type=int)
    dataset.add_argument("--probe-seeds", dest="probe_seeds", nargs="+", type=int)
    dataset.add_argument("--no-resume", action="store_true")
    dataset.set_defaults(func=_cmd_dataset)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
