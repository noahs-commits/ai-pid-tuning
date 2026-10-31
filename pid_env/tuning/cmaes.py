"""CMA-ES tuning for PID gains on arbitrary environment configurations."""

from __future__ import annotations

import concurrent.futures
import math
import multiprocessing
from typing import Callable, Sequence

import numpy as np

from ..config import EnvironmentConfig
from ..evaluation import (
    LossHyperparameters,
    PIDSettings,
    evaluate,
)
from .serialization import (
    cmaes_config_from_dict,
    environment_config_from_dict,
    environment_config_to_dict,
    loss_hyperparameters_from_dict,
    loss_hyperparameters_to_dict,
    pid_settings_to_dict,
)
from .types import CMAESConfig, CMAESResult


def encode_gains(
    settings: PIDSettings,
    config: CMAESConfig,
) -> np.ndarray:
    low, high = config.gain_bounds
    gains = np.clip(
        np.asarray([settings.kp, settings.ki, settings.kd], dtype=np.float64),
        low,
        high,
    )
    if config.log_space:
        return np.log10(gains)
    return gains


def decode_gains(x: Sequence[float], config: CMAESConfig) -> PIDSettings:
    values = np.asarray(x, dtype=np.float64)
    if config.log_space:
        values = np.power(10.0, values)
    low, high = config.gain_bounds
    values = np.clip(values, low, high)
    return PIDSettings(
        kp=float(values[0]),
        ki=float(values[1]),
        kd=float(values[2]),
        integral_limits=config.initial_gains.integral_limits,
    )


def _evaluate_encoded(
    x: Sequence[float],
    *,
    env_config: EnvironmentConfig,
    hyperparameters: LossHyperparameters,
    seeds: Sequence[int],
    config: CMAESConfig,
) -> float:
    return evaluate(
        decode_gains(x, config),
        env_config,
        hyperparameters,
        seeds=seeds,
    )


def _evaluate_worker(args: tuple) -> float:
    x, env_config_dict, hyper_dict, seeds, config_dict = args
    config = cmaes_config_from_dict(config_dict)
    return _evaluate_encoded(
        x,
        env_config=environment_config_from_dict(env_config_dict),
        hyperparameters=loss_hyperparameters_from_dict(hyper_dict),
        seeds=seeds,
        config=config,
    )


def _encoded_bounds(config: CMAESConfig) -> np.ndarray:
    low, high = config.gain_bounds
    if config.log_space:
        return np.asarray([math.log10(low), math.log10(high)], dtype=np.float64)
    return np.asarray([low, high], dtype=np.float64)


def _run_search(
    es,
    config: CMAESConfig,
    budget: int,
    evaluate_population: Callable[[list], list[float]],
    *,
    restart: int,
    history: list[dict],
) -> tuple[np.ndarray | None, float | None, int, int]:
    best_x: np.ndarray | None = None
    best_loss: float | None = None
    evaluations = 0
    generations = 0

    while evaluations < budget and not es.stop():
        if evaluations > 0 and budget - evaluations < es.popsize:
            break
        solutions = es.ask()
        fitnesses = evaluate_population(solutions)
        es.tell(solutions, fitnesses)

        finite = [
            (value, index)
            for index, value in enumerate(fitnesses)
            if math.isfinite(value)
        ]
        if finite:
            generation_best, index = min(finite, key=lambda item: item[0])
            if best_loss is None or generation_best < best_loss:
                best_loss = generation_best
                best_x = np.asarray(solutions[index], dtype=np.float64).copy()

            history.append(
                {
                    "restart": restart,
                    "generation": generations,
                    "evaluations": evaluations + len(solutions),
                    "best_loss": float(best_loss),
                    "generation_best": float(generation_best),
                    "mean_loss": float(np.mean([value for value, _ in finite])),
                    "best_gains": pid_settings_to_dict(
                        decode_gains(best_x, config)
                    ),
                }
            )

        evaluations += len(solutions)
        generations += 1

    return best_x, best_loss, evaluations, generations


def tune(
    env_config: EnvironmentConfig,
    hyperparameters: LossHyperparameters,
    config: CMAESConfig | None = None,
    *,
    callback: Callable[[dict], None] | None = None,
) -> CMAESResult:
    """Find PID gains for ``env_config`` with CMA-ES.

    Each candidate is scored on the same fixed set of ``config.seeds``.  A
    separate validation seed set is only used to report the final result.
    """
    config = config or CMAESConfig()

    try:
        import cma
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "CMA-ES tuning requires the 'cma' package; install it with "
            "`pip install cma` or `pip install -e .[tuning]`"
        ) from exc

    if env_config.thrust <= env_config.weight:
        raise ValueError(
            "environment is infeasible: thrust must be greater than weight"
        )

    x0 = encode_gains(config.initial_gains, config)
    bounds = _encoded_bounds(config)
    x0 = np.clip(x0, bounds[0], bounds[1])

    env_config_dict = environment_config_to_dict(env_config)
    hyper_dict = loss_hyperparameters_to_dict(hyperparameters)
    config_dict = {
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

    executor = None
    if config.workers > 1:
        # Prefer fork on POSIX: it lets tune() be called from a plain script
        # without a __main__ guard.  Platforms without fork (Windows, macOS
        # spawn) still require the guard in the caller.
        try:
            mp_context = multiprocessing.get_context("fork")
        except ValueError:
            mp_context = None
        executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=config.workers,
            mp_context=mp_context,
        )

    def evaluate_population(solutions: list) -> list[float]:
        if executor is None:
            return [
                _evaluate_encoded(
                    x,
                    env_config=env_config,
                    hyperparameters=hyperparameters,
                    seeds=config.seeds,
                    config=config,
                )
                for x in solutions
            ]
        payloads = [
            (x, env_config_dict, hyper_dict, config.seeds, config_dict)
            for x in solutions
        ]
        return list(executor.map(_evaluate_worker, payloads))

    total_best_x: np.ndarray | None = None
    total_best_loss: float | None = None
    total_evaluations = 0
    total_generations = 0
    history: list[dict] = []

    runs = config.restarts + 1
    budget_per_run = max(1, config.max_evaluations // runs)

    try:
        for restart in range(runs):
            options = {
                "bounds": [
                    [float(bounds[0])] * 3,
                    [float(bounds[1])] * 3,
                ],
                "seed": config.seed + restart,
                "verbose": config.verbose,
                "maxfevals": budget_per_run,
                "tolfun": config.tolfun,
            }
            if config.popsize is not None:
                options["popsize"] = config.popsize

            es = cma.CMAEvolutionStrategy(x0, config.sigma0, options)
            best_x, best_loss, evaluations, generations = _run_search(
                es,
                config,
                budget_per_run,
                evaluate_population,
                restart=restart,
                history=history,
            )
            total_evaluations += evaluations
            total_generations += generations
            if best_loss is not None and (
                total_best_loss is None or best_loss < total_best_loss
            ):
                total_best_loss = best_loss
                total_best_x = best_x
            if callback is not None and history:
                callback(history[-1])
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    if total_best_x is None or total_best_loss is None:
        raise RuntimeError("CMA-ES did not evaluate any candidates")

    best_gains = decode_gains(total_best_x, config)
    validation_loss = evaluate(
        best_gains,
        env_config,
        hyperparameters,
        seeds=config.validation_seeds,
    )

    return CMAESResult(
        best_gains=best_gains,
        best_loss=float(total_best_loss),
        validation_loss=float(validation_loss),
        evaluations=total_evaluations,
        generations=total_generations,
        history=history,
        env_config=env_config,
        hyperparameters=hyperparameters,
        cma_config=config,
    )
