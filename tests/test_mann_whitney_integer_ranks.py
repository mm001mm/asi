"""Public rank comparisons must retain exact integer ordering and ties."""

import math

import numpy as np
import pytest

from alberta_framework.utils.statistics import mann_whitney_comparison


@pytest.mark.parametrize("dtype", [np.int64, np.uint64])
def test_large_integer_samples_keep_exact_order(dtype):
    # All eight A values precede all eight B values. Two of the 16-choose-8
    # permutations are this extreme, independent of the shared integer offset.
    a = np.asarray([2**60 + i for i in range(8)], dtype=dtype)
    b = np.asarray([2**60 + i for i in range(8, 16)], dtype=dtype)
    result = mann_whitney_comparison(a, b)
    assert result.statistic == 0.0
    assert result.p_value == pytest.approx(2 / math.comb(16, 8))
    assert result.significant
    assert result.effect_size == -1.0
    reversed_result = mann_whitney_comparison(b, a)
    assert reversed_result.statistic == 64.0
    assert reversed_result.p_value == result.p_value
    assert reversed_result.effect_size == 1.0


def test_mixed_signed_unsigned_constants_are_not_false_ties():
    a = np.full(8, 2**60, dtype=np.int64)
    b = np.full(8, 2**60 + 1, dtype=np.uint64)
    result = mann_whitney_comparison(a, b)
    reference = mann_whitney_comparison(np.zeros(8), np.ones(8))
    assert result == reference
    assert result.significant
    assert result.effect_size == -1.0


def test_mixed_integer_samples_preserve_actual_ties():
    # The reduced samples have the same complete ordering and tie groups.
    a = np.asarray([2**60 + i for i in [0, 0, 1, 2]], dtype=np.int64)
    b = np.asarray([2**60 + i for i in [1, 2, 3, 3]], dtype=np.uint64)
    result = mann_whitney_comparison(a, b)
    reference = mann_whitney_comparison([0, 0, 1, 2], [1, 2, 3, 3])
    assert result == reference


def test_negative_integer_samples_preserve_order():
    a = np.asarray([-2**60 + i for i in range(8)], dtype=np.int64)
    b = np.asarray([-2**60 + i for i in range(8, 16)], dtype=np.int64)
    result = mann_whitney_comparison(a, b)
    assert result.statistic == 0.0
    assert result.p_value == pytest.approx(2 / math.comb(16, 8))
    assert result.significant


def test_ordinary_integer_and_float_comparisons_agree():
    a = np.asarray([3, 1, 4, 1, 5], dtype=np.int64)
    b = np.asarray([2, 7, 1, 8, 2], dtype=np.uint64)
    assert mann_whitney_comparison(a, b) == mann_whitney_comparison(
        a.astype(np.float64), b.astype(np.float64)
    )
