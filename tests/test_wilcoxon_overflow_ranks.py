"""Signed-rank inference must not turn distinct finite-pair gaps into tied infinities."""

import itertools
from collections.abc import Sequence
from fractions import Fraction
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from alberta_framework.utils.export import generate_significance_table
from alberta_framework.utils.statistics import wilcoxon_comparison

pytestmark = pytest.mark.unit


def exact_signed_rank_result(
    a: Sequence[float | int], b: Sequence[float | int]
) -> tuple[float, float]:
    """Enumerate the sign null using exact differences of the supplied numbers."""
    differences = [Fraction(x) - Fraction(y) for x, y in zip(a, b)]
    nonzero = [value for value in differences if value]
    magnitudes = [abs(value) for value in nonzero]
    # Twice the average rank is integer, including ties. Count rather than
    # sort/rank so this oracle does not reuse the production implementation.
    weights = [
        2 * sum(other < value for other in magnitudes)
        + sum(other == value for other in magnitudes) + 1
        for value in magnitudes
    ]
    total = sum(weights)
    positive = sum(weight for weight, value in zip(weights, nonzero) if value > 0)
    observed = min(positive, total - positive)
    extreme = sum(
        min(selected, total - selected) <= observed
        for signs in itertools.product((0, 1), repeat=len(weights))
        for selected in [sum(sign * weight for sign, weight in zip(signs, weights))]
    )
    return observed / 2, extreme / (2 ** len(weights))


@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_overflowing_differences_preserve_default_significance(sign: float) -> None:
    small = np.array([80, 82, 84, -90, 92, 94, 96, 98, 100.0]) * sign
    large = small * 1e306
    expected = exact_signed_rank_result(large.tolist(), (-large).tolist())
    assert expected == (4.0, 0.02734375)
    control = wilcoxon_comparison(small, -small)
    result = wilcoxon_comparison(large, -large)
    assert (control.statistic, control.p_value) == expected
    assert (result.statistic, result.p_value) == expected
    assert result.significant is True
    assert result.effect_size == pytest.approx(control.effect_size)
    for table_format in ("markdown", "latex"):
        table = generate_significance_table({("A", "B"): result}, format=table_format)
        assert "0.0273" in table
        assert "Yes" in table


def test_overflow_fallback_keeps_ties_zero_and_subnormal_differences() -> None:
    smallest = np.nextafter(0.0, 1.0)
    a = [1e308, -1e308, 9.5e307, -9.5e307, 8e307, -8e307, smallest, 0.0]
    b = [-1e308, 1e308, -9.5e307, 9.5e307, -8e307, 8e307, 0.0, 0.0]
    expected = exact_signed_rank_result(a, b)
    result = wilcoxon_comparison(a, b)
    assert (result.statistic, result.p_value) == expected


def test_finite_difference_control_uses_the_same_rank_null() -> None:
    a = [1.0, 2.0, -2.0, 4.0, -5.0, 0.0]
    b = [0.0] * len(a)
    expected = exact_signed_rank_result(a, b)
    result = wilcoxon_comparison(a, b)
    assert (result.statistic, result.p_value) == expected


@pytest.mark.parametrize("dtype", [np.int64, np.uint64])
def test_integer_pair_gaps_survive_a_large_shared_offset(dtype: type) -> None:
    a: NDArray[Any] = np.asarray([2**60 + i for i in range(1, 9)], dtype=dtype)
    b: NDArray[Any] = np.full(8, 2**60, dtype=dtype)
    expected = exact_signed_rank_result(a.tolist(), b.tolist())
    assert expected == (0.0, 0.0078125)
    result = wilcoxon_comparison(a, b)
    assert (result.statistic, result.p_value) == expected
    assert result.significant


def test_mixed_integer_constants_keep_their_nonzero_pair_gaps() -> None:
    a: NDArray[Any] = np.full(8, 2**60, dtype=np.int64)
    b: NDArray[Any] = np.full(8, 2**60 + 1, dtype=np.uint64)
    expected = exact_signed_rank_result(a.tolist(), b.tolist())
    assert expected == (0.0, 0.0078125)
    result = wilcoxon_comparison(a, b)
    assert (result.statistic, result.p_value) == expected
    assert result.significant
    for table_format in ("markdown", "latex"):
        table = generate_significance_table({("A", "B"): result}, format=table_format)
        assert "0.0078" in table
        assert "Yes" in table


def test_mixed_integer_pair_gaps_preserve_zero_and_actual_ties() -> None:
    a: NDArray[Any] = np.asarray([2**60 + i for i in [-3, -2, -1, 0, 1, 2, 3, 4]], dtype=np.int64)
    b: NDArray[Any] = np.full(8, 2**60, dtype=np.uint64)
    expected = exact_signed_rank_result(a.tolist(), b.tolist())
    result = wilcoxon_comparison(a, b)
    assert (result.statistic, result.p_value) == expected


def test_integer_pair_gaps_preserve_variation_beyond_float64_precision() -> None:
    a: NDArray[Any] = np.asarray(
        [2**62 + i if i != 6 else -2**62 - i for i in range(0, 18, 2)],
        dtype=np.int64,
    )
    b: NDArray[Any] = np.asarray(
        [-2**62 if i != 6 else 2**62 for i in range(0, 18, 2)], dtype=np.int64
    )
    expected = exact_signed_rank_result(a.tolist(), b.tolist())
    assert expected == (4.0, 0.02734375)
    result = wilcoxon_comparison(a, b)
    assert (result.statistic, result.p_value) == expected


@pytest.mark.parametrize("dtype", [np.int8, np.uint8])
def test_small_integer_pairs_keep_the_existing_signed_rank_result(dtype: type) -> None:
    a: NDArray[Any] = np.asarray([10, 11, 12, 0, 13, 14, 15, 16, 17], dtype=dtype)
    b: NDArray[Any] = np.asarray([0, 0, 0, 20, 0, 0, 0, 0, 0], dtype=dtype)
    expected = exact_signed_rank_result(a.tolist(), b.tolist())
    result = wilcoxon_comparison(a, b)
    assert (result.statistic, result.p_value) == expected
