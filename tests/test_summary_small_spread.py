"""A common offset must not distort an exactly representable sample spread."""

import itertools
from collections.abc import Sequence
from decimal import Decimal, localcontext
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy import stats

from alberta_framework.utils.statistics import (
    bootstrap_ci,
    compute_statistics,
    compute_timeseries_statistics,
)

pytestmark = pytest.mark.unit


def decimal_summary(values: Sequence[float | int]) -> tuple[float, float, float, float, float]:
    """Independent oracle over exact integer or binary64 input values."""
    with localcontext() as context:
        context.prec = 100
        numbers = [Decimal(value) for value in values]
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


@pytest.mark.parametrize("samples", [
    np.array([2**60, 2**60 + 1, 2**60 + 2], dtype=np.int64),
    np.array([2**63 + 63, 2**63 + 64, 2**63 + 65], dtype=np.uint64),
    np.array([-2**60 - 2, -2**60 - 1, -2**60], dtype=np.int64),
    np.array([2**63 - 3, 2**63 - 2, 2**63 - 1], dtype=np.int64),
    np.array([2**60 + 63, 2**60 + 127, 2**60 + 191], dtype=np.int64),
    np.array([2**53 + 1, 2**53 + 2, 2**53 + 3], dtype=np.uint64),
])
def test_integer_summary_keeps_exact_local_spread(samples: NDArray[Any]) -> None:
    numbers = [int(value) for value in samples]
    mean, sd, sem, lower, upper = decimal_summary(numbers)
    expected_iqr = (max(numbers) - min(numbers)) / 2
    for permutation in itertools.permutations(range(3)):
        result = compute_statistics(samples[list(permutation)])
        assert result.mean == mean
        assert result.std == pytest.approx(sd, rel=2e-15, abs=0.0)
        assert result.sem == pytest.approx(sem, rel=2e-15, abs=0.0)
        assert result.median == float(sorted(numbers)[1])
        assert result.iqr == expected_iqr
        assert (result.ci_lower, result.ci_upper) == (lower, upper)


def test_integer_timeseries_keeps_exact_local_spread() -> None:
    values = np.array([[2**60 + offset, -2**60 - offset] for offset in [63, 127, 191]])
    expected = [decimal_summary([int(value) for value in column]) for column in values.T]
    mean, lower, upper = compute_timeseries_statistics(values)
    np.testing.assert_array_equal(mean, [row[0] for row in expected])
    np.testing.assert_array_equal(lower, [row[3] for row in expected])
    np.testing.assert_array_equal(upper, [row[4] for row in expected])


@pytest.mark.parametrize("sign", [-1, 1])
def test_integer_bootstrap_keeps_local_spread_before_rounding(sign: int) -> None:
    values: NDArray[Any] = np.array(
        [sign * (2**60 + offset) for offset in [63, 127, 191]], dtype=np.int64
    )
    numbers = [Decimal(int(value)) for value in values]
    rng = np.random.default_rng(42)
    with localcontext() as context:
        context.prec = 100
        expected_point = float(sum(numbers) / 3)
        sampled = sorted(sum((numbers[i] for i in rng.choice(3, size=3)), Decimal(0)) / 3
                         for _ in range(81))
        expected = (expected_point, float(sampled[20]), float(sampled[60]))
    assert bootstrap_ci(values, confidence_level=0.5, n_bootstrap=81, seed=42) == expected
