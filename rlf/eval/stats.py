"""Confidence intervals and sample sizes. Normal quantiles come from `statistics.NormalDist`."""
from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist

import numpy as np


@dataclass(frozen=True)
class Estimate:
    mean: float
    lo: float
    hi: float
    n: int
    level: float = 0.95


def _z(level: float) -> float:
    return NormalDist().inv_cdf(0.5 + level / 2)


def mean_ci(x: np.ndarray, level: float = 0.95) -> Estimate:
    """Normal-approximation CI of the mean, standard error with ddof=1."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    m = float(x.mean())
    se = float(x.std(ddof=1)) / math.sqrt(n)
    z = _z(level)
    return Estimate(m, m - z * se, m + z * se, n, level)


def proportion_ci(successes: int, n: int, level: float = 0.95) -> Estimate:
    """Wilson score interval."""
    z = _z(level)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return Estimate(p, max(0.0, centre - half), min(1.0, centre + half), n, level)


def paired_diff_ci(a: np.ndarray, b: np.ndarray, level: float = 0.95) -> Estimate:
    """CI of the per-episode difference a - b (same seeds in both arrays)."""
    return mean_ci(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64), level)


def normalized_score_ci(agent, rand, greedy, alpha: float, level: float = 0.95) -> Estimate:
    """CI of mean(d_i), d_i = a_i - (1-alpha) r_i - alpha g_i; the claim "score >= alpha" passes if lo >= 0."""
    a, r, g = (np.asarray(v, dtype=np.float64) for v in (agent, rand, greedy))
    return mean_ci(a - (1 - alpha) * r - alpha * g, level)


def sample_size_proportion(p: float, half_width: float, level: float = 0.95) -> int:
    return math.ceil(_z(level) ** 2 * p * (1 - p) / half_width ** 2)


def sample_size_mean(sigma: float, half_width: float, level: float = 0.95) -> int:
    return math.ceil((_z(level) * sigma / half_width) ** 2)


class InsufficientSample(ValueError):
    """Raised when a claim is attempted with fewer than the required independent samples."""


def require_n(n: int, minimum: int = 1000) -> None:
    if n < minimum:
        raise InsufficientSample(f"n={n} < {minimum}: not enough samples to make a claim")


def passes_lower(est: Estimate, threshold: float) -> bool:
    """Claim "X >= threshold": passes if the lower CI bound is >= threshold (needs n >= 1000)."""
    require_n(est.n)
    return est.lo >= threshold
