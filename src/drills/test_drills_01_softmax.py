import math

import numpy as np
import pytest
from drills_01_softmax import cross_entropy, log_softmax, softmax


@pytest.mark.parametrize(
    "np_input",
    [
        np.random.rand(4, 7),
        np.array([1000, 1001, 1002], dtype=np.int32),
        np.array([-1000, -1001, -1002], dtype=np.int32),
        np.random.rand(2, 3, 5),
    ],
    ids=[
        "Random 2-D array",
        "Large positive array",
        "Large negative array",
        "Random 3-D array",
    ],
)
def test_softmax_for_input(np_input):
    # Test
    result = softmax(np_input)

    # Assert
    sums = np.sum(result, axis=-1)
    expected = np.ones_like(sums)
    np.testing.assert_allclose(sums, expected, rtol=1e-09, atol=1e-12)


@pytest.mark.parametrize("axis", [0, 1, 2, -1])
def test_softmax_sums_to_one_along_axis(axis):
    # Arrange
    np_input = np.random.rand(2, 3, 5)

    # Test
    result = softmax(np_input, axis=axis)

    # Assert
    sums = result.sum(axis=axis)
    np.testing.assert_allclose(sums, np.ones_like(sums), rtol=1e-09, atol=1e-12)


@pytest.mark.parametrize(
    "np_input",
    [np.array([1000, 1001, 1002]), np.random.rand(4, 7)],
    ids=["Overflow test", "Random 2-D array"],
)
def test_log_softmax_for_input(np_input):
    # Test
    result = log_softmax(np_input)

    # Assert
    exp_result = np.exp(result)
    inp_softmax = softmax(np_input)
    np.testing.assert_allclose(exp_result, inp_softmax, rtol=1e-09, atol=1e-12)


@pytest.mark.parametrize("axis", [0, 1, 2, -1])
def test_log_softmax_on_axis(axis):
    # Arrange
    np_input = np.random.rand(2, 3, 5)

    # Test
    result = log_softmax(np_input, axis)

    # Assert
    exp_result = np.exp(result)
    inp_softmax = softmax(np_input, axis)
    np.testing.assert_allclose(exp_result, inp_softmax, rtol=1e-09, atol=1e-12)


@pytest.mark.parametrize(
    "logits, target, preds",
    [
        (
            np.array([[0, 1, 2], [0, 1, 2]], dtype=np.int32),
            np.array([2, 0], dtype=np.int32),
            math.log(1 + math.exp(-1) + math.exp(-2)) + 1,
        ),
        (np.array([[100, 0, 0]], dtype=np.int32), np.array([0], dtype=np.int32), 0.0),
        (
            np.array([[0, 0, 0, 0]], dtype=np.int32),
            np.array([2], dtype=np.int32),
            math.log(4),
        ),
    ],
)
def test_cross_entropy(logits, target, preds):
    # Test
    result = cross_entropy(logits, target)

    # Assert
    np.testing.assert_allclose(result, preds, rtol=1e-09, atol=1e-12)
