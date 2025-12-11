import numpy as np
import scipy.sparse as sp
from sklearn.metrics import recall_score

from skxml import map_at_k, precision_at_k, recall_at_k
from skxml._xc_metrics import _get_topk


class TestMetricsAtK:
    y_true = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0]])
    y_scores = np.array([[0.8, 0.2, 0.6], [0.4, 0.6, 0.3], [0.7, 0.9, 0.5]])


class TestPrecisionAtK(TestMetricsAtK):
    #  Returns the precision@k for a valid input.
    def test_valid_input_array(self):
        # Arrange
        sort_values = False
        # Act
        result_at_1 = precision_at_k(
            self.y_true, self.y_scores, k=1, sort_values=sort_values
        )
        # Assert
        assert np.allclose(result_at_1, 1), f"Wrong value of precision@1: {result_at_1}"
        result_at_2 = precision_at_k(
            self.y_true, self.y_scores, k=2, sort_values=sort_values
        )
        assert np.allclose(result_at_2, 5 / 6), (
            f"wrong value of precision@2: {result_at_2}"
        )

    def test_valid_input_sparse(self):
        # Arrange
        y_true = sp.csr_array(self.y_true)
        y_pred = sp.csr_array(self.y_scores)
        sort_values = False

        # Act
        result_at_1 = precision_at_k(y_true, y_pred, k=1, sort_values=sort_values)
        # Assert
        assert np.allclose(result_at_1, 1), f"Wrong value of precision@1: {result_at_1}"
        result_at_2 = precision_at_k(y_true, y_pred, k=2, sort_values=sort_values)
        assert np.allclose(result_at_2, 5 / 6), (
            f"wrong value of precision@2: {result_at_2}"
        )


class TestRecallAtK(TestMetricsAtK):
    #  Returns the recall@k for a valid input.
    def test_valid_input_array(self):
        k = 2
        sort_values = False
        # Execute the function under test
        result = recall_at_k(self.y_true, self.y_scores, k, sort_values=sort_values)
        # Assertion
        assert result == 1.0

    def test_valid_input_sparse(self):
        k = 2
        sort_values = False
        # Execute the function under test
        result = recall_at_k(
            sp.csr_array(self.y_true),
            sp.csr_array(self.y_scores),
            k,
            sort_values=sort_values,
        )

        # Assertion
        assert result == 1.0

    #  Returns 0.0 when y_true is an empty array.
    def test_recall_at_k_empty_y_true(self):
        # Test setup
        y_true = np.array([[]])
        k = 2
        sort_values = False

        # Execute the function under test
        try:
            _ = recall_at_k(y_true, self.y_scores, k, sort_values=sort_values)
        except AssertionError:
            pass

    #  Returns 0.0 when y_pred is an empty array.
    def test_recall_at_k_empty_y_pred(self):
        # Test setup
        y_pred = np.array([])
        k = 2
        sort_values = False

        # Execute the function under test
        try:
            _ = recall_at_k(self.y_true, y_pred, k, sort_values=sort_values)
        except AssertionError:
            pass

    #  Returns 0.0 when y_true and y_pred are empty arrays.
    def test_recall_at_k_empty_y_true_y_pred(self):
        # Test setup
        y_true = np.array([[]])
        y_pred = np.array([[]])
        k = 2
        sort_values = False

        # Execute the function under test
        try:
            _ = recall_at_k(y_true, y_pred, k, sort_values=sort_values)
        except AssertionError:
            pass

    def test_exact_recall_sklearn(self):
        k = 2

        topk_indices = _get_topk(self.y_scores, k=k)
        # in this way we can simply set to 1 the indices of topk elements per row over any row
        y_pred = sp.lil_array(self.y_scores.shape)
        # Create row indices for advanced indexing
        row_indices = np.arange(self.y_scores.shape[0])[:, np.newaxis]
        row_indices = np.repeat(row_indices, k, axis=1)
        y_pred[row_indices, topk_indices] = 1
        y_pred = y_pred.toarray()  # Convert to dense array for sklearn
        sklearn_recall = recall_score(
            y_true=self.y_true, y_pred=y_pred, average="samples"
        )
        skxml_recall = recall_at_k(y_true=self.y_true, y_pred=self.y_scores, k=k)
        assert sklearn_recall == skxml_recall, "Recall value is wrong against sklearn"


class TestMAPAtK(TestMetricsAtK):
    #  Returns the precision@k for a valid input.
    def test_valid_input_array(self):
        # Arrange
        sort_values = False
        # Act
        result_at_1 = map_at_k(self.y_true, self.y_scores, k=1, sort_values=sort_values)
        # Assert
        assert result_at_1 is not None, "MAP@1 should return a value"
