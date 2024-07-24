import numpy as np
import scipy.sparse as sp
from numpy.typing import ArrayLike
from scipy.special import softmax
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    hamming_loss,
    jaccard_score,
    log_loss,
    precision_score,
    recall_score,
    zero_one_loss,
)

from skxml._xc_metrics import (
    compute_inv_propensity,
    ndcg,
    precision,
    psndcg,
    psprecision,
    psrecall,
    recall,
    meanavgprecision,
)


def precision_at_k(
    y_true: np.ndarray | sp.csr_matrix | sp.csr_array,
    y_pred: np.ndarray | sp.csr_matrix | sp.csr_array,
    k: int = 1,
    propensity_array: np.ndarray | None = None,
    propensity_coeff: tuple[float, float] | None = None,
    sort_values: bool = False,
) -> float:
    """
    Returns the precision@k.

    Parameters
    ----------
    y_true: np.ndarray, sp.csr_matrix, dict
        The 2D array of ground truth labels.
    y_pred: sp.csr_matrix, np.ndarray or dict
        The 2D array of labels relevance as found by the classifier .predict_proba
        * sp.csr_matrix: sp.csr_matrix with nnz at relevant places
        * np.ndarray (float): scores for each label
            User must ensure shape is fine
        * np.ndarray (int): top indices (in sort_values order)
            User must ensure shape is fine
        * {'indices': np.ndarray, 'scores': np.ndarray}
    k: int
        The number of indices to return.
    propensity_array:
        An array with the inverse propensity scores
    propensity_coeff:
        A tuple with two elements representing the propensity coefficients
    sort_values:
        whether to sort values
    """
    if isinstance(propensity_coeff, tuple | list):
        propensity_array = compute_inv_propensity(
            labels=y_true, A=propensity_coeff[0], B=propensity_coeff[1]
        )
    if isinstance(propensity_array, np.ndarray):
        return psprecision(
            X=y_pred,
            true_labels=y_true,
            inv_psp=propensity_array,
            k=k,
            sort_values=sort_values,
        )[-1]
    elif propensity_array is None:
        return float(
            precision(X=y_pred, true_labels=y_true, k=k, sort_values=sort_values)[-1]
        )
    else:
        raise ValueError("Unsupported propensity array type")


def mean_average_precision_at_k(
    y_true: np.ndarray | sp.csr_matrix | sp.csr_array,
    y_pred: np.ndarray | sp.csr_matrix | sp.csr_array,
    k: int = 1,
    propensity_array: np.ndarray | None = None,
    propensity_coeff: tuple[float, float] | None = None,
    sort_values: bool = False,
) -> float:
    """
    Returns the mean average precision@k.

    Parameters
    ----------
    y_true: np.ndarray, sp.csr_matrix, dict
        The 2D array of ground truth labels.
    y_pred: sp.csr_matrix, np.ndarray or dict
        The 2D array of labels relevance as found by the classifier .predict_proba
        * sp.csr_matrix: sp.csr_matrix with nnz at relevant places
        * np.ndarray (float): scores for each label
            User must ensure shape is fine
        * np.ndarray (int): top indices (in sort_values order)
            User must ensure shape is fine
        * {'indices': np.ndarray, 'scores': np.ndarray}
    k: int
        The number of indices to return.
    propensity_array:
        An array with the inverse propensity scores
    propensity_coeff:
        A tuple with two elements representing the propensity coefficients
    sort_values:
        whether to sort values
    """
    if isinstance(propensity_coeff, tuple | list):
        propensity_array = compute_inv_propensity(
            labels=y_true, A=propensity_coeff[0], B=propensity_coeff[1]
        )
    if isinstance(propensity_array, np.ndarray):
        return np.mean(
            psprecision(
                X=y_pred,
                true_labels=y_true,
                inv_psp=propensity_array,
                k=k,
                sort_values=sort_values,
            )
            / np.arange(1, k)
        )
    elif propensity_array is None:
        return float(
            np.mean(
                precision(X=y_pred, true_labels=y_true, k=k, sort_values=sort_values)
                / np.arange(1, k)
            )
        )
    else:
        raise ValueError("Unsupported propensity array type")


def f1_at_k(
    y_true: np.ndarray | sp.csr_matrix | sp.csr_array,
    y_pred: np.ndarray | sp.csr_matrix | sp.csr_array,
    k: int = 1,
    propensity_array: np.ndarray | None = None,
    propensity_coeff: tuple[float, float] | None = None,
    sort_values: bool = False,
) -> float:
    """
    Returns the f1@k.

    Parameters
    ----------
    y_true: np.ndarray, sp.csr_matrix, dict
        The 2D array of ground truth labels.
    y_pred: sp.csr_matrix, np.ndarray or dict
        The 2D array of labels relevance as found by the classifier .predict_proba
        * sp.csr_matrix: sp.csr_matrix with nnz at relevant places
        * np.ndarray (float): scores for each label
            User must ensure shape is fine
        * np.ndarray (int): top indices (in sort_values order)
            User must ensure shape is fine
        * {'indices': np.ndarray, 'scores': np.ndarray}
    k: int
        The number of indices to return.
    propensity_array:
        An array with the inverse propensity scores
    propensity_coeff:
        A tuple with two elements representing the propensity coefficients
    sort_values:
        whether to s
    """
    p = precision_at_k(
        y_true, y_pred, k, propensity_array, propensity_coeff, sort_values
    )
    r = recall_at_k(y_true, y_pred, k, propensity_array, propensity_coeff, sort_values)
    return 2 * p * r / (p + r)


def recall_at_k(
    y_true: np.ndarray | sp.csr_matrix | sp.csr_array,
    y_pred: np.ndarray | sp.csr_matrix | sp.csr_array,
    k: int = 1,
    propensity_array: np.ndarray | None = None,
    propensity_coeff: tuple[float, float] | None = None,
    sort_values: bool = False,
) -> float:
    """
    Returns the recall@k.

    Parameters
    ----------
    y_true: np.ndarray, sp.csr_matrix, dict
        The 2D array of ground truth labels.
    y_pred: sp.csr_matrix, np.ndarray or dict
        The 2D array of labels relevance as found by the classifier .predict_proba
        * sp.csr_matrix: sp.csr_matrix with nnz at relevant places
        * np.ndarray (float): scores for each label
            User must ensure shape is fine
        * np.ndarray (int): top indices (in sort_values order)
            User must ensure shape is fine
        * {'indices': np.ndarray, 'scores': np.ndarray}

    k: int
        The number of indices to return.
    propensity_array:
        An array with the inverse propensity scores
    propensity_coeff:
        A tuple with two elements representing the propensity coefficients
    sort_values: bool
    """
    if isinstance(propensity_coeff, tuple | list):
        propensity_array = compute_inv_propensity(
            labels=y_true, A=propensity_coeff[0], B=propensity_coeff[1]
        )
    if isinstance(propensity_array, np.ndarray):
        return float(
            psrecall(
                X=y_pred,
                true_labels=y_true,
                inv_psp=propensity_array,
                k=k,
                sort_values=sort_values,
            )[-1]
        )
    elif propensity_array is None:
        return float(
            recall(X=y_pred, true_labels=y_true, k=k, sort_values=sort_values)[-1]
        )
    else:
        raise ValueError("Unsupported propensity array type")


def ndcg_at_k(
    y_true: np.ndarray | sp.csr_matrix | sp.csr_array,
    y_pred: np.ndarray | sp.csr_matrix | sp.csr_array,
    k: int = 1,
    propensity_array: np.ndarray | None = None,
    propensity_coeff: tuple[float, float] = None,
    sort_values: bool = False,
) -> float:
    """
    Returns the normalized cumulative discount gain@k.

    Parameters
    ----------
    y_true: np.ndarray, sp.csr_matrix, dict
        The 2D array of ground truth labels.
    y_pred: sp.csr_matrix, np.ndarray or dict
        The 2D array of labels relevance as found by the classifier .predict_proba
        * sp.csr_matrix: sp.csr_matrix with nnz at relevant places
        * np.ndarray (float): scores for each label
            User must ensure shape is fine
        * np.ndarray (int): top indices (in sort_values order)
            User must ensure shape is fine
        * {'indices': np.ndarray, 'scores': np.ndarray}

    k: int
        The number of indices to return.
    propensity_array:
        An array with the inverse propensity scores
    propensity_coeff:
        A tuple with two elements representing the propensity coefficients
    sort_values: bool
    """
    if isinstance(propensity_coeff, tuple | list):
        propensity_array = compute_inv_propensity(
            labels=y_true, A=propensity_coeff[0], B=propensity_coeff[1]
        )
    if isinstance(propensity_array, np.ndarray):
        return float(
            psndcg(
                X=y_pred,
                true_labels=y_true,
                inv_psp=propensity_array,
                k=k,
                sort_values=sort_values,
            )[-1]
        )
    elif propensity_array is None:
        return float(
            ndcg(X=y_pred, true_labels=y_true, k=k, sort_values=sort_values)[-1]
        )
    else:
        raise ValueError("Unsupported propensity array type")


def validate_shapes(y_true, y_pred):
    if y_pred is not None:
        if y_true.shape != y_pred.shape:
            raise ValueError("Incompatbile shapes between arrays")


def validate_types(y_true, y_score, y_pred):
    if y_true.dtype != np.int64:
        raise ValueError("Convert y_true array to int")
    if y_pred.dtype != np.int64 and y_pred.dtype != bool:
        raise ValueError("Convert y_pred array to int")
    if y_score is not None:
        if y_score.dtype != np.float64:
            raise ValueError("Convert y_score array to float")


def compute_metrics(
    y_true: ArrayLike | sp.sparray,
    y_pred: ArrayLike | sp.sparray,
    y_score: ArrayLike | None = None,
    sample_weight: ArrayLike | None = None,
    **kwargs,
) -> dict[str, float]:
    """
    Computes the metrics for the model evaluation.

    Parameters
    ----------
    y_true: np.ndarray, scipy.sparse sparse array, pd.DataFrame
        The true labels.
    y_pred: np.ndarray, sp.csr_matrix, pd.DataFrame
        The predicted labels.
    y_score: np.ndarary, sparse array, pd.DataFrame
        The predicted scores as from the method .predict_proba
    sample_weight: Optional[ArrayLike]
        The weight to pass to each individual sample when performing
        averaged versions of the metrics.
    Returns
    -------
    Dict
        A dictionary containing the metrics.
    """

    validate_shapes(y_true, y_pred)
    validate_shapes(y_true, y_score)
    validate_shapes(y_pred, y_score)

    # validate_types(y_true, y_pred, y_score)

    all_metrics = {
        "precision_weighted": precision_score(
            y_true=y_true,
            y_pred=y_pred,
            average="weighted",
            zero_division=0,
            sample_weight=sample_weight,
        ),
        "precision_micro": precision_score(
            y_true=y_true,
            y_pred=y_pred,
            average="micro",
            zero_division=0,
            sample_weight=sample_weight,
        ),
        "recall_weighted": recall_score(
            y_true=y_true,
            y_pred=y_pred,
            average="weighted",
            zero_division=0,
            sample_weight=sample_weight,
        ),
        "recall_micro": recall_score(
            y_true=y_true,
            y_pred=y_pred,
            average="micro",
            zero_division=0,
            sample_weight=sample_weight,
        ),
        "f1_weighted": f1_score(
            y_true=y_true,
            y_pred=y_pred,
            average="weighted",
            zero_division=0,
            sample_weight=sample_weight,
        ),
        "f1_micro": f1_score(
            y_true=y_true,
            y_pred=y_pred,
            average="micro",
            zero_division=0,
            sample_weight=sample_weight,
        ),
        "accuracy": accuracy_score(
            y_true=y_true, y_pred=y_pred, sample_weight=sample_weight
        ),
        "hamming_loss": hamming_loss(
            y_true=y_true, y_pred=y_pred, sample_weight=sample_weight
        ),
        "zero_one_loss": zero_one_loss(
            y_true=y_true, y_pred=y_pred, sample_weight=sample_weight
        ),
        "jaccard_score_weighted": jaccard_score(
            y_true=y_true,
            y_pred=y_pred,
            average="weighted",
            zero_division=0.0,
            sample_weight=sample_weight,
        ),
        "jaccard_score_micro": jaccard_score(
            y_true=y_true,
            y_pred=y_pred,
            average="micro",
            zero_division=0.0,
            sample_weight=sample_weight,
        ),
    }

    if y_score is not None:
        if sp.issparse(y_score):
            y_score = y_score.toarray()
        if "k" in kwargs:
            K = kwargs["k"]
            propensity_coeff = kwargs.get("propensity_coeff", None)
            for k in range(1, K + 1):
                all_metrics[f"ncdg@{k}"] = ndcg_at_k(y_true=y_true, y_pred=y_score, k=k)
                all_metrics[f"ncdg@{k}"] = ndcg_at_k(y_true=y_true, y_pred=y_score, k=k)
                all_metrics[f"precision@{k}"] = precision_at_k(
                    y_true=y_true, y_pred=y_score, k=k
                )
                all_metrics[f"map@{k}"] = mean_average_precision_at_k(
                    y_true=y_true,
                    y_pred=y_score,
                    k=k,
                    propensity_coeff=propensity_coeff,
                )
                all_metrics[f"recall@{k}"] = recall_at_k(
                    y_true=y_true, y_pred=y_score, k=k
                )
                all_metrics[f"f1@{k}"] = (
                    2
                    * all_metrics[f"recall@{k}"]
                    * all_metrics[f"precision@{k}"]
                    / (all_metrics[f"recall@{k}"] + all_metrics[f"precision@{k}"])
                )
                # when propensity coefficients also the propensity-scored metrics are computed
                if propensity_coeff is not None:
                    all_metrics[f"psncdg@{k}"] = ndcg_at_k(
                        y_true=y_true,
                        y_pred=y_score,
                        k=k,
                        propensity_coeff=propensity_coeff,
                    )
                    all_metrics[f"psprecision@{k}"] = precision_at_k(
                        y_true=y_true,
                        y_pred=y_score,
                        k=k,
                        propensity_coeff=propensity_coeff,
                    )
                    all_metrics[f"psrecall@{k}"] = recall_at_k(
                        y_true=y_true,
                        y_pred=y_score,
                        k=k,
                        propensity_coeff=propensity_coeff,
                    )
                    all_metrics[f"psf1@{k}"] = (
                        2
                        * all_metrics[f"psrecall@{k}"]
                        * all_metrics[f"psprecision@{k}"]
                        / (
                            all_metrics[f"psrecall@{k}"]
                            + all_metrics[f"psprecision@{k}"]
                        )
                    )

    if y_score is not None:
        all_metrics["log_loss"] = log_loss(
            y_true=y_true,
            y_pred=softmax(y_score, axis=1),
            sample_weight=sample_weight,
        )
    all_metrics["support"] = y_pred.shape[0]

    return all_metrics
