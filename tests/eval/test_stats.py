import math

import numpy as np
import pytest

from rlf.eval.stats import (Estimate, InsufficientSample, mean_ci, normalized_score_ci,
                            paired_diff_ci, passes_lower, proportion_ci, require_n,
                            sample_size_mean, sample_size_proportion)

Z = 1.959963984540054


def test_mean_ci_known_values():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    e = mean_ci(x)
    se = x.std(ddof=1) / math.sqrt(5)
    assert e.mean == 3.0 and e.n == 5 and e.level == 0.95
    assert e.lo == pytest.approx(3.0 - Z * se) and e.hi == pytest.approx(3.0 + Z * se)


def test_mean_ci_level_widens():
    x = np.random.default_rng(0).normal(size=200)
    assert mean_ci(x, 0.99).hi - mean_ci(x, 0.99).lo > mean_ci(x, 0.95).hi - mean_ci(x, 0.95).lo


def test_wilson_known_value():
    e = proportion_ci(50, 100)
    assert e.mean == 0.5 and e.n == 100
    assert e.lo == pytest.approx(0.40383, abs=1e-4) and e.hi == pytest.approx(0.59617, abs=1e-4)


def test_wilson_edges_stay_in_unit_interval():
    e0, e1 = proportion_ci(0, 50), proportion_ci(50, 50)
    assert e0.lo == 0.0 and 0 < e0.hi < 0.2
    assert e1.hi == pytest.approx(1.0) and 0.8 < e1.lo < 1


def test_wilson_coverage():
    rng = np.random.default_rng(1)
    hits = 0
    for _ in range(2000):
        k = int(rng.binomial(1000, 0.6))
        e = proportion_ci(k, 1000)
        hits += e.lo <= 0.6 <= e.hi
    assert 0.93 <= hits / 2000 <= 0.97


def test_paired_diff_is_mean_ci_of_difference():
    rng = np.random.default_rng(2)
    a, b = rng.normal(size=100), rng.normal(size=100)
    assert paired_diff_ci(a, b) == mean_ci(a - b)


def test_paired_beats_unpaired_when_correlated():
    rng = np.random.default_rng(3)
    base = rng.normal(size=1000) * 5
    a, b = base + rng.normal(size=1000) * 0.1 + 0.3, base
    paired = paired_diff_ci(a, b)
    assert (paired.hi - paired.lo) < 0.2 * (mean_ci(a).hi - mean_ci(a).lo)


def test_normalized_score_endpoints():
    rng = np.random.default_rng(4)
    r, g = rng.normal(0, 1, 1000), rng.normal(5, 1, 1000)
    a = 0.5 * r + 0.5 * g
    e = normalized_score_ci(a, r, g, alpha=0.5)
    assert e.mean == pytest.approx(0.0, abs=1e-12) and e.n == 1000
    e1 = normalized_score_ci(g, r, g, alpha=1.0)
    assert e1.mean == pytest.approx(0.0, abs=1e-12)
    e_better = normalized_score_ci(g + 1.0, r, g, alpha=1.0)
    assert e_better.mean == pytest.approx(1.0) and e_better.lo > 0


def test_sample_sizes():
    assert sample_size_proportion(0.5, 0.03) == 1068
    assert sample_size_mean(2.0, 0.1) == math.ceil((Z * 2.0 / 0.1) ** 2)
    assert sample_size_proportion(0.5, 0.01) > sample_size_proportion(0.5, 0.03)


def test_require_n_and_passes_lower():
    with pytest.raises(InsufficientSample):
        require_n(999)
    require_n(1000)
    require_n(50, minimum=10)
    ok = Estimate(mean=1.0, lo=0.9, hi=1.1, n=1000)
    assert passes_lower(ok, 0.9) and not passes_lower(ok, 0.91)
    with pytest.raises(InsufficientSample):
        passes_lower(Estimate(mean=1.0, lo=0.9, hi=1.1, n=999), 0.0)
    assert issubclass(InsufficientSample, ValueError)
