"""CMA-ES tuning and dataset-generation utilities."""

from .cmaes import decode_gains, encode_gains, tune
from .dataset import build_shard, generate_dataset
from .probe import probe_rollout, sample_probe
from .sampler import (
    ENV_FEATURE_NAMES,
    env_feature_vector,
    sample_environment,
    sample_prior_gains,
)
from .types import (
    CMAESConfig,
    CMAESResult,
    DatasetConfig,
    EnvSamplerConfig,
    GainSamplerConfig,
    ProbeConfig,
)

__all__ = [
    "CMAESConfig",
    "CMAESResult",
    "DatasetConfig",
    "ENV_FEATURE_NAMES",
    "EnvSamplerConfig",
    "GainSamplerConfig",
    "ProbeConfig",
    "build_shard",
    "decode_gains",
    "encode_gains",
    "env_feature_vector",
    "generate_dataset",
    "probe_rollout",
    "sample_environment",
    "sample_prior_gains",
    "sample_probe",
    "tune",
]
