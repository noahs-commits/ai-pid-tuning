"""One-dimensional fractal (octave) Perlin noise."""

from __future__ import annotations

import numpy as np


def _fade(t: np.ndarray) -> np.ndarray:
    """Perlin's quintic smoothing curve."""
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


class FractalPerlinNoise:
    """Seeded fractal Perlin noise in one dimension.

    A single octave of 1D gradient noise produces a smooth signal in roughly
    ``[-1, 1]``.  Several octaves are summed with geometrically decreasing
    amplitude (``persistence``) and increasing frequency (``lacunarity``),
    then normalised back into ``[-1, 1]``.

    Parameters
    ----------
    seed:
        Seed for the permutation table and per-octave offsets.
    octaves:
        Number of noise layers.  More octaves add finer detail.
    persistence:
        Amplitude multiplier per octave (0 < persistence < 1).
    lacunarity:
        Frequency multiplier per octave (> 1).
    frequency:
        Base frequency applied to the input coordinate.
    """

    def __init__(
        self,
        seed: int = 0,
        octaves: int = 4,
        persistence: float = 0.5,
        lacunarity: float = 2.0,
        frequency: float = 1.0,
    ) -> None:
        if octaves < 1:
            raise ValueError("octaves must be >= 1")
        if not 0.0 < persistence <= 1.0:
            raise ValueError("persistence must be in (0, 1]")
        if lacunarity <= 0.0:
            raise ValueError("lacunarity must be > 0")
        if frequency <= 0.0:
            raise ValueError("frequency must be > 0")

        self.seed = int(seed)
        self.octaves = int(octaves)
        self.persistence = float(persistence)
        self.lacunarity = float(lacunarity)
        self.frequency = float(frequency)

        rng = np.random.default_rng(self.seed)
        permutation = rng.permutation(256).astype(np.int64)
        self._perm = np.concatenate([permutation, permutation])
        self._offsets = rng.uniform(0.0, 256.0, size=self.octaves)

    def _perlin(self, x: np.ndarray) -> np.ndarray:
        """One octave of 1D gradient noise."""
        x = np.asarray(x, dtype=np.float64)
        xi = np.floor(x).astype(np.int64)
        xf = x - xi
        xi &= 255

        g0 = np.where(self._perm[xi] & 1, 1.0, -1.0)
        g1 = np.where(self._perm[xi + 1] & 1, 1.0, -1.0)

        n0 = g0 * xf
        n1 = g1 * (xf - 1.0)
        u = _fade(xf)
        return 2.0 * (n0 + u * (n1 - n0))

    def sample(self, x: float | np.ndarray) -> np.ndarray:
        """Evaluate the fractal noise at coordinate(s) ``x``.

        Returns a float array (or numpy scalar for scalar input) in
        ``[-1, 1]``.
        """
        x = np.asarray(x, dtype=np.float64)
        total = np.zeros_like(x, dtype=np.float64)
        amplitude = 1.0
        frequency = self.frequency
        max_amplitude = 0.0

        for octave in range(self.octaves):
            total += amplitude * self._perlin(x * frequency + self._offsets[octave])
            max_amplitude += amplitude
            amplitude *= self.persistence
            frequency *= self.lacunarity

        return np.clip(total / max_amplitude, -1.0, 1.0)

    __call__ = sample
