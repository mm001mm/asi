"""Signed-rank inference must not turn distinct finite-pair gaps into tied infinities."""

import itertools
from fractions import Fraction

import numpy as np
import pytest

from alberta_framework.utils.export import generate_significance_table
from alberta_framework.utils.statistics import wilcoxon_comparison

pytestmark = pytest.mark.unit


def exact_signed_rank_result(a: list[float], b: list[float]) -> tuple[float, float]:
    """Enumerate the two-sided sign null using exact differences of binary64 inputs."""
    differences = [Fraction.from_float(x) - Fraction.from_float(y) for x, y in zip(a, b)]
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
