"""
Pytest suite to validate functionality of distortions and evaluator.
(ARTwarp DTW gets tested in artwarp-py, assumed to be valid).

@author: Jaime Barquero Cebrian
"""

import pytest
from pytest_check import check
from pathlib import Path
import numpy as np
import warping


def test_csvLoader():
    freq = warping.csvLoader(
        Path(__file__).parent.parent / "data" / "F109-2001-SW-IND63104_ROCCA.csv"
    )
    with check:
        assert freq[0] == 4218.75
    with check:
        assert freq.shape == (212,)
    with check:
        assert freq.dtype == np.float64
    with check:
        assert freq.min() > 0


@pytest.mark.parametrize(
    "freq, expected",
    [
        ([10, 20, 30, 40, 50], [10, 16.6667, 23.3333, 30, 50]),
        ([1, 2, 3, 4, 5], [1, 1.66667, 2.333, 3, 5]),
    ],
)
def test_xDistortion(freq, expected):
    planGraph, distorted, key, _ = warping.xDistortion(freq, 2, 20)
    planGraph_expected = [0, 0.6667, 1.3333, 2, 4]
    key_expected = [0, 1.5, 3, 3.5, 4]

    _, _, key_shift_0, _ = warping.xDistortion(freq, 2, 0)

    np.testing.assert_allclose(distorted, expected, atol=5e-2)
    np.testing.assert_allclose(planGraph, planGraph_expected, atol=5e-2)
    np.testing.assert_allclose(key, key_expected, atol=5e-2)
    np.testing.assert_allclose(key_shift_0, np.arange(len(freq)), atol=5e-2)


anchor_shift_data = [(0, 2), (2, -4), (2, 10**99), (10**99, 0)]


@pytest.mark.parametrize("anchor, shift", anchor_shift_data)
def test_xDistortion_raises(anchor, shift):
    freq = warping.csvLoader(
        Path(__file__).parent.parent / "data" / "F109-2001-SW-IND63104_ROCCA.csv"
    )

    with pytest.raises(ValueError):
        warping.xDistortion(freq, anchor, shift)


@pytest.mark.parametrize(
    "freq, expected", [([10, 10, 10, 10, 10], [10.67667, 13.03265, 15, 13.03265, 10.67667])]
)
def test_yDistortion(freq, expected):
    # High value for height has been chosen to exagerate bump
    bumped = warping.yDistortion(freq, 2, 2, 20, 0.5)

    bumped_no_width = warping.yDistortion(freq, 2, 2, None, 0.5)
    no_width_expected = [10, 10.019, 15, 10.019, 10]

    np.testing.assert_allclose(bumped, expected, atol=5e-2)
    np.testing.assert_allclose(bumped_no_width, no_width_expected, atol=5e-2)
    with pytest.raises(ValueError):
        warping.yDistortion(freq, 2, 1, -20, 0.3)


def testDistortion():
    freq = warping.csvLoader(
        Path(__file__).parent.parent / "data" / "F109-2001-SW-IND63104_ROCCA.csv"
    )

    _, x_distort, _, _ = warping.xDistortion(freq, 100, 20)
    y_distort = warping.yDistortion(freq, 100, 100, 20, 0.15)

    distort_no_height = warping.distort(freq, 100, 20, 0, 20)
    distort_no_shift = warping.distort(freq, 100, 20, 0.15, 0)

    np.testing.assert_allclose(x_distort, distort_no_height[0])
    np.testing.assert_allclose(y_distort, distort_no_shift[0])
