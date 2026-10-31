"""A common offset must not distort an exactly representable sample spread."""

import itertools
from decimal import Decimal, localcontext

import numpy as np
import pytest
from scipy import stats

from alberta_framework.utils.statistics import (
    bootstrap_ci,
    compute_statistics,
    compute_timeseries_statistics,
)

pytestmark = pytest.mark.unit


def decimal_summary(values: list[float]) -> tuple[float, float, float, float, float]:
    """Independent high-precision oracle over the actual binary64 input values."""
    with localcontext() as context:
        context.prec = 100
        numbers = [Decimal.from_float(value) for value in values]
        mean = sum(numbers, Decimal(0)) / Decimal(len(numbers))
        variance = sum(((value - mean) ** 2 for value in numbers), Decimal(0)) / Decimal(
            len(numbers) - 1
        )
        sd = variance.sqrt()
        sem = (variance / Decimal(len(numbers))).sqrt()
        critical = Decimal.from_float(float(stats.t.ppf(0.975, len(numbers) - 1)))
        return (
            float(mean), float(sd), float(sem),
            float(mean - critical * sem), float(mean + critical * sem),
        )


@pytest.mark.parametrize("sign", [-1.0, 1.0])
@pytest.mark.parametrize("values", [[1e16, 1e16 + 2, 1e16 + 4], [1e16, 1e16 + 2]])
def test_summary_retains_representable_small_spread(values: list[float], sign: float) -> None:
    samples = [sign * value for value in values]
    expected_mean, expected_sd, expected_sem, lower, upper = decimal_summary(samples)
    for permutation in itertools.permutations(samples):
        result = compute_statistics(list(permutation))
        assert result.mean == expected_mean
        assert result.std == pytest.approx(expected_sd, rel=2e-15, abs=0.0)
        assert result.sem == pytest.approx(expected_sem, rel=2e-15, abs=0.0)
        assert result.ci_lower == lower
        assert result.ci_upper == upper


def test_timeseries_retains_small_spread_in_each_column() -> None:
    values = np.array(
        [[1e16, -1e16], [1e16 + 2, -1e16 - 2], [1e16 + 4, -1e16 - 4]]
    )
    expected = [decimal_summary(column.tolist()) for column in values.T]
    mean, lower, upper = compute_timeseries_statistics(values)
    np.testing.assert_array_equal(mean, [row[0] for row in expected])
    np.testing.assert_array_equal(lower, [row[3] for row in expected])
    np.testing.assert_array_equal(upper, [row[4] for row in expected])


def test_centering_keeps_ordinary_and_extreme_constant_controls() -> None:
    result = compute_statistics([1.0, 2.0, 3.0, 4.0, 5.0])
    assert result.mean == 3.0
    assert result.std == pytest.approx(np.sqrt(2.5))
    maximum = np.finfo(np.float64).max
    for value in [maximum, -maximum, 0.0]:
        result = compute_statistics([value, value])
        assert result.mean == value
        assert result.std == result.sem == 0.0
        assert result.ci_lower == result.ci_upper == value


def test_subnormal_mean_still_rounds_only_at_the_result() -> None:
    smallest = np.nextafter(0.0, 1.0)
    values = np.array([smallest, 2 * smallest])
    # The exact mean is 1.5 * smallest: round-to-even gives 2 * smallest.
    assert compute_statistics(values).mean == 2 * smallest
    mean, _, _ = compute_timeseries_statistics(values[:, None])
    assert mean[0] == 2 * smallest


@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_bootstrap_mean_retains_small_spread(sign: float) -> None:
    values = [sign * value for value in [1e16, 1e16 + 2, 1e16 + 4]]
    rng = np.random.default_rng(42)
    with localcontext() as context:
        context.prec = 100
        numbers = [Decimal.from_float(value) for value in values]
        expected_point = float(sum(numbers, Decimal(0)) / 3)
        sampled_means = sorted(
            sum((numbers[index] for index in rng.choice(3, size=3)), Decimal(0)) / 3
            for _ in range(81)
        )
        # At 50% confidence with 81 samples, the percentile ranks are exactly
        # 20 and 60. No percentile implementation is shared with this oracle.
        expected = (expected_point, float(sampled_means[20]), float(sampled_means[60]))
    assert bootstrap_ci(values, confidence_level=0.5, n_bootstrap=81, seed=42) == expected
