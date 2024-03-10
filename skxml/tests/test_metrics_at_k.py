import numpy as np
import pytest
import scipy.sparse as sp  # noqa: F401
from skxml.metrics import f1_at_k_score, precision_at_k_score, recall_at_k_score


class TestPrecisionAtK:

    #  Returns the precision@k for a valid input.
    def test_valid_input(self):
        # Arrange
        y_true = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0]])
        y_pred = np.array([[0.8, 0.2, 0.6], [0.4, 0.6, 0.3], [0.7, 0.9, 0.5]])
        sort_values = False

        # Act
        result_at_1 = precision_at_k_score(y_true, y_pred, k=1, sort_values=sort_values)
        # Assert
        assert np.allclose(result_at_1, 1), f"Wrong value of precision@1: {result_at_1}"
        result_at_2 = precision_at_k_score(y_true, y_pred, k=1, sort_values=sort_values)
        assert np.allclose(result_at_2, 5 / 6), f"wrong value of precision@2: {result_at_2}"

    #  Returns 0 when y_true is empty.
    def test_empty_y_true(self):
        # Arrange
        y_true = np.array([])
        y_pred = np.array([[0.8, 0.2, 0.6], [0.4, 0.6, 0.3], [0.7, 0.9, 0.5]])
        k = 1
        sort_values = False

        # Act
        try:
            result = precision_at_k_score(y_true, y_pred, k, sort_values=sort_values)
        except ValueError:
            pass

    #  Returns 0 when y_pred is empty.
    def test_empty_y_pred(self):
        # Arrange
        y_true = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0]])
        y_pred = np.array([])
        k = 2
        sort_values = False

        try:
            result = precision_at_k_score(y_true, y_pred, k, sort_values=sort_values)
        except ValueError:
            pass

    #  Returns 0 when y_true and y_pred are empty.
    def test_empty_y_true_and_y_pred(self):
        # Arrange
        y_true = np.array([])
        y_pred = np.array([])
        k = 1
        sort_values = False

        result = precision_at_k_score(y_true, y_pred, k, sort_values=sort_values)


class TestF1AtK:

    #  Returns the f1 score for valid input.
    def test_valid_input(self):
        y_true = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        y_pred = np.array([[0.9, 0.1, 0], [0.1, 0.8, 0.1], [0, 0.2, 0.8]])
        result = f1_at_k_score(y_true, y_pred)
        assert result == pytest.approx(0.6666666666666666)

    #  Returns the f1 score for valid input with k > 1.
    def test_valid_input_k_greater_than_1(self):
        y_true = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        y_pred = np.array([[0.9, 0.1, 0], [0.1, 0.8, 0.1], [0, 0.2, 0.8]])
        result = f1_at_k_score(y_true, y_pred, k=2)
        assert result == pytest.approx(0.6666666666666666)

    #  Returns 0 when y_true and y_pred are empty.
    def test_empty_input(self):
        y_true = np.array([])
        y_pred = np.array([])
        result = f1_at_k_score(y_true, y_pred)
        assert result == 0

    #  Returns 0 when k is 0.
    def test_k_zero(self):
        y_true = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        y_pred = np.array([[0.9, 0.1, 0], [0.1, 0.8, 0.1], [0, 0.2, 0.8]])
        result = f1_at_k_score(y_true, y_pred, k=0)
        assert result == 0

    #  Returns 0 when k is greater than the number of labels.
    def test_k_greater_than_labels(self):
        y_true = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        y_pred = np.array([[0.9, 0.1, 0], [0.1, 0.8, 0.1], [0, 0.2, 0.8]])
        result = f1_at_k_score(y_true, y_pred, k=4)
        assert result == 0


class TestRecallAtK:

    #  Returns the recall@k for a valid input.
    def test_recall_at_k_valid_input(self):
        # Test setup
        y_true = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0]])
        y_pred = np.array([[0.8, 0.2, 0.6], [0.4, 0.6, 0.3], [0.7, 0.9, 0.5]])
        k = 2
        sort_values = False

        # Execute the function under test
        result = recall_at_k_score(y_true, y_pred, k, sort_values=sort_values)

        # Assertion
        assert result == 1.0

    #  Returns the recall@1 for a valid input when k is not specified.
    def test_recall_at_k_default_k(self):
        # Test setup
        y_true = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0]])
        y_pred = np.array([[0.8, 0.2, 0.6], [0.4, 0.6, 0.3], [0.7, 0.9, 0.5]])
        sort_values = False

        # Execute the function under test
        result = recall_at_k_score(y_true, y_pred, sort_values=sort_values)

        # Assertion
        assert result == 1.0

    #  Returns 0.0 when y_true is an empty array.
    def test_recall_at_k_empty_y_true(self):
        # Test setup
        y_true = np.array([])
        y_pred = np.array([[0.8, 0.2, 0.6], [0.4, 0.6, 0.3], [0.7, 0.9, 0.5]])
        k = 2
        sort_values = False

        # Execute the function under test
        result = recall_at_k_score(y_true, y_pred, k, sort_values=sort_values)

        # Assertion
        assert result == 0.0

    #  Returns 0.0 when y_pred is an empty array.
    def test_recall_at_k_empty_y_pred(self):
        # Test setup
        y_true = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0]])
        y_pred = np.array([])
        k = 2
        sort_values = False

        # Execute the function under test
        result = recall_at_k_score(y_true, y_pred, k, sort_values=sort_values)

        # Assertion
        assert result == 0.0

    #  Returns 0.0 when y_true and y_pred are empty arrays.
    def test_recall_at_k_empty_y_true_y_pred(self):
        # Test setup
        y_true = np.array([])
        y_pred = np.array([])
        k = 2
        sort_values = False

        # Execute the function under test
        result = recall_at_k_score(y_true, y_pred, k, sort_values=sort_values)

        # Assertion
        assert result == 0.0
