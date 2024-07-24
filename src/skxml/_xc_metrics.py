from typing import Union, Tuple

import numba as nb
import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_array, csr_matrix


SparseLike = Union[csr_array, csr_matrix]
ArrayLike = np.typing.ArrayLike
SparseOrDenseLike = Union[SparseLike, ArrayLike]


@nb.njit(parallel=True)
def _topk_nb(
    data: SparseLike,
    indices: np.ndarray,
    indptr: np.ndarray,
    k: int,
    pad_ind: np.ndarray,
    pad_val: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get top-k indices and values for a sparse (csr) matrix
    * Parallel version: uses numba
    Arguments:
    ---------
    data: np.ndarray
        data / vals of csr array
    indices: np.ndarray
        indices of csr array
    indptr: np.ndarray
        indptr of csr array
    k: int
        values to select
    pad_ind: int
        padding index for indices array
        Useful when number of values in a row are less than k
    pad_val: int
        padding index for values array
        Useful when number of values in a row are less than k
    Returns:
    --------
    ind: np.ndarray
        topk indices; size=(num_rows, k)
    val: np.ndarray, optional
        topk val; size=(num_rows, k)
    """
    nr = len(indptr) - 1
    ind = np.full((nr, k), fill_value=pad_ind, dtype=indices.dtype)
    val = np.full((nr, k), fill_value=pad_val, dtype=data.dtype)

    for i in nb.prange(nr):
        s, e = indptr[i], indptr[i + 1]
        num_el = min(k, e - s)
        temp = np.argsort(data[s:e])[::-1][:num_el]
        ind[i, :num_el] = indices[s:e][temp]
        val[i, :num_el] = data[s:e][temp]
    return ind, val


def topk(
    X: SparseLike,
    k: int,
    pad_ind: int,
    pad_val: int,
    return_values=False,
    dtype="float32",
):
    """
    Get top-k indices and values for a sparse (csr) matrix
    Arguments:
    ---------
    X: csr_matrix
        sparse matrix
    k: int
        values to select
    pad_ind: int
        padding index for indices array
        Useful when number of values in a row are less than k
    pad_val: int
        padding index for values array
        Useful when number of values in a row are less than k
    return_values: boolean, optional, default=False
        Return topk values or not
    dtype: str, optional, default='float32'
        datatype of values

    Returns:
    --------
    ind: np.ndarray
        topk indices; size=(num_rows, k)
    val: np.ndarray, optional
        topk val; size=(num_rows, k)
    """
    ind, val = _topk_nb(X.data, X.indices, X.indptr, k, pad_ind, pad_val)
    if return_values:
        return ind, val.astype(dtype)
    else:
        return ind


def compatible_shapes(x: SparseOrDenseLike, y: SparseOrDenseLike):
    """
    See if both matrices have same shape

    Works fine for the following combinations:
    * both are sparse
    * both are dense

    Will only compare rows when:
    * one is sparse/dense and other is dict
    * one is sparse and other is dense

    ** User must ensure that predictions are of correct shape when a
    np.ndarray is passed with all predictions.
    """
    # both are either sparse or dense
    if (sp.issparse(x) and sp.issparse(y)) or (
        isinstance(x, np.ndarray) and isinstance(y, np.ndarray)
    ):
        return x.shape == y.shape

    # compare #rows if one is sparse and other is dict or np.ndarray
    if not (isinstance(x, dict) or isinstance(y, dict)):
        return x.shape[0] == y.shape[0]
    else:
        if isinstance(x, dict):
            return len(x["indices"]) == len(x["scores"]) == y.shape[0]
        else:
            return len(y["indices"]) == len(y["scores"]) == x.shape[0]


def _get_topk_sparse(X: SparseLike, pad_indx: int = 0, k=5):
    """
    Get top-k elements when X is a sparse matrix
    """
    X = X.tocsr()
    X.sort_indices()
    pad_indx = X.shape[1]
    indices = topk(X, k, pad_indx, 0, return_values=False)
    return indices


def _get_topk_array(X: ArrayLike, k: int = 5, sort_values: bool = False):
    """
    Get top-k elements when X is an array
    X can be an array of:
        indices: indices of top predictions (must be sorted)
        values: scores for all labels (like in one-vs-all)
    """
    # indices are given
    assert X.shape[1] >= k, "Number of elements in X is < {}".format(k)
    if np.issubdtype(X.dtype, np.integer):
        assert sorted, "sorted must be true with indices"
        indices = X[:, :k] if X.shape[1] > k else X
    # values are given
    elif np.issubdtype(X.dtype, np.floating):
        _indices = np.argpartition(X, -k)[:, -k:]
        _scores = np.take_along_axis(X, _indices, axis=-1)
        indices = np.argsort(-_scores, axis=-1)
        indices = np.take_along_axis(_indices, indices, axis=1)
    return indices


def _get_topk_dict(X: dict, k: int = 5, sort_values: bool = False):
    """
    Get top-k elements when X is an dict of indices and scores
    X['scores'][i, j] will contain score of
        ith instance and X['indices'][i, j]th label
    """
    indices = X["indices"]
    scores = X["scores"]
    assert compatible_shapes(
        indices, scores
    ), "Dimension mis-match: expected array of shape {} found {}".format(
        indices.shape, scores.shape
    )
    assert scores.shape[1] >= k, "Number of elements in X is < {}".format(k)
    # assumes indices are already sorted by the user
    if sort_values:
        return indices[:, :k] if indices.shape[1] > k else indices

    # get top-k entried without sorting them
    if scores.shape[1] > k:
        _indices = np.argpartition(scores, -k)[:, -k:]
        _scores = np.take_along_axis(scores, _indices, axis=-1)
        # sort top-k entries
        __indices = np.argsort(-_scores, axis=-1)
        _indices = np.take_along_axis(_indices, __indices, axis=-1)
        indices = np.take_along_axis(indices, _indices, axis=-1)
    else:
        _indices = np.argsort(-scores, axis=-1)
        indices = np.take_along_axis(indices, _indices, axis=-1)
    return indices


def _get_topk(X: SparseOrDenseLike, pad_indx: int = 0, k=5, sort_values: bool = False):
    """
    Get top-k indices (row-wise); Support for
    * csr_matirx
    * 2 np.ndarray with indices and values
    * np.ndarray with indices or values
    """
    if sp.issparse(X):
        indices = _get_topk_sparse(X=X, pad_indx=pad_indx, k=k)
    elif isinstance(X, np.ndarray):
        indices = _get_topk_array(X=X, k=k, sort_values=sort_values)
    elif isinstance(X, dict):
        indices = _get_topk_dict(X=X, k=k, sort_values=sort_values)
    else:
        raise NotImplementedError(
            "Unknown type; please pass csr_matrix, np.ndarray or dict."
        )
    return indices


def compute_inv_propensity(labels: SparseLike, A: float, B: float) -> np.ndarray:
    """
    Computes inverse propernsity as proposed in Jain et al. 16.

    Arguments:
    ---------
    labels: csr_matrix
        label matrix (typically ground truth for train data)
    A: float
        typical values:
        * 0.5: Wikipedia
        * 0.6: Amazon
        * 0.55: otherwise
    B: float
        typical values:
        * 0.4: Wikipedia
        * 2.6: Amazon
        * 1.5: otherwise

    Returns:
    -------
    np.ndarray: propensity scores for each label
    """
    num_instances, _ = labels.shape
    freqs = np.ravel(np.sum(labels, axis=0))
    C = (np.log(num_instances) - 1) * np.power(B + 1, A)
    wts = 1.0 + C * np.power(freqs + B, -A)
    return np.ravel(wts)


def _setup_metric(X, true_labels, inv_psp=None, k=5, sort_values: bool = False):
    def _broad_cast(mat, like):
        if isinstance(like, np.ndarray):
            return np.asarray(mat)
        elif sp.issparse(mat):
            return mat
        else:
            raise NotImplementedError(
                "Unknown type; please pass csr_matrix, np.ndarray or dict."
            )

    assert compatible_shapes(
        X, true_labels
    ), "ground truth and prediction matrices must have same shape."
    num_instances, num_labels = true_labels.shape
    indices = _get_topk(X, num_labels, k, sort_values)
    ps_indices = None
    if inv_psp is not None:
        _mat = sp.spdiags(inv_psp, diags=0, m=num_labels, n=num_labels)
        _psp_wtd = _broad_cast(_mat.dot(true_labels.T).T, true_labels)
        ps_indices = _get_topk(_psp_wtd, num_labels, k, False)
        inv_psp = np.hstack([inv_psp, np.zeros((1))])

    idx_dtype = true_labels.indices.dtype
    true_labels = sp.csr_matrix(
        (true_labels.data, true_labels.indices, true_labels.indptr),
        shape=(num_instances, num_labels + 1),
        dtype=true_labels.dtype,
    )

    # scipy won't respect the dtype of indices
    # may fail otherwise on really large datasets
    true_labels.indices = true_labels.indices.astype(idx_dtype)
    return indices, true_labels, ps_indices, inv_psp


def _eval_flags(indices, true_labels, inv_psp=None):
    if sp.issparse(true_labels):
        nr, nc = indices.shape
        rows = np.repeat(np.arange(nr).reshape(-1, 1), nc)
        eval_flags = true_labels[rows, indices.ravel()].A1.reshape(nr, nc)
    elif type(true_labels) == np.ndarray:
        eval_flags = np.take_along_axis(true_labels, indices, axis=-1)
    if inv_psp is not None:
        eval_flags = np.multiply(inv_psp[indices], eval_flags)
    return eval_flags


def precision(X, true_labels, k=5, sort_values=False):
    """
    Compute precision@k for 1-k

    Arguments:
    ----------
    X: csr_matrix, np.ndarray or dict
        * csr_matrix: csr_matrix with nnz at relevant places
        * np.ndarray (float): scores for each label
            User must ensure shape is fine
        * np.ndarray (int): top indices (in sorted order)
            User must ensure shape is fine
        * {'indices': np.ndarray, 'scores': np.ndarray}

    true_labels: csr_matrix or np.ndarray
        ground truth in sparse or dense format
    k: int, optional (default=5)
        compute precision till k
    sorted: boolean, optional, default=False
        whether X is already sorted (will skip sorting)
        * used when X is of type dict or np.ndarray (of indices)
        * shape is not checked is X are np.ndarray
        * must be set to true when X are np.ndarray (of indices)
    Returns:
    -------
    np.ndarray: precision values for 1-k
    """
    indices, true_labels, _, _ = _setup_metric(
        X, true_labels, k=k, sort_values=sort_values
    )
    eval_flags = _eval_flags(indices, true_labels, None)
    return _precision(eval_flags, k)


def psprecision(X, true_labels, inv_psp, k=5, sort_values=False):
    """
    Compute propensity scored precision@k for 1-k

    Arguments:
    ----------
    X: csr_matrix, np.ndarray or dict
        * csr_matrix: csr_matrix with nnz at relevant places
        * np.ndarray (float): scores for each label
            User must ensure shape is fine
        * np.ndarray (int): top indices (in sorted order)
            User must ensure shape is fine
        * {'indices': np.ndarray, 'scores': np.ndarray}
    true_labels: csr_matrix or np.ndarray
        ground truth in sparse or dense format
    inv_psp: np.ndarray
        propensity scores for each label
    k: int, optional (default=5)
        compute propensity scored precision till k
    sorted: boolean, optional, default=False
        whether X is already sorted (will skip sorting)
        * used when X is of type dict or np.ndarray (of indices)
        * shape is not checked is X are np.ndarray
        * must be set to true when X are np.ndarray (of indices)


    Returns:
    -------
    np.ndarray: propensity scored precision values for 1-k
    """
    indices, true_labels, ps_indices, inv_psp = _setup_metric(
        X, true_labels, inv_psp, k=k, sort_values=sort_values
    )
    eval_flags = _eval_flags(indices, true_labels, inv_psp)
    ps_eval_flags = _eval_flags(ps_indices, true_labels, inv_psp)
    return _precision(eval_flags, k) / _precision(ps_eval_flags, k)


def _precision(eval_flags, k=5):
    deno = 1 / (np.arange(k) + 1)
    precision = np.mean(np.multiply(np.cumsum(eval_flags, axis=-1), deno), axis=0)
    return np.ravel(precision)


def ndcg(X, true_labels, k=5, sort_values=False):
    """
    Compute nDCG@k for 1-k

    Arguments:
    ----------
    X: csr_matrix, np.ndarray or dict
        * csr_matrix: csr_matrix with nnz at relevant places
        * np.ndarray (float): scores for each label
            User must ensure shape is fine
        * np.ndarray (int): top indices (in sorted order)
            User must ensure shape is fine
        * {'indices': np.ndarray, 'scores': np.ndarray}
    true_labels: csr_matrix or np.ndarray
        ground truth in sparse or dense format
    k: int, optional (default=5)
        compute nDCG till k
    sorted: boolean, optional, default=False
        whether X is already sorted (will skip sorting)
        * used when X is of type dict or np.ndarray (of indices)
        * shape is not checked is X are np.ndarray
        * must be set to true when X are np.ndarray (of indices)


    Returns:
    -------
    np.ndarray: nDCG values for 1-k
    """
    indices, true_labels, _, _ = _setup_metric(
        X, true_labels, k=k, sort_values=sort_values
    )
    eval_flags = _eval_flags(indices, true_labels, None)
    _total_pos = np.asarray(true_labels.sum(axis=1), dtype=np.int32)
    _max_pos = max(np.max(_total_pos), k)
    _cumsum = np.cumsum(1 / np.log2(np.arange(1, _max_pos + 1) + 1))
    n = _cumsum[_total_pos - 1]
    return _ndcg(eval_flags, n, k)


def psndcg(X, true_labels, inv_psp, k=5, sort_values=False):
    """
    Compute propensity scored nDCG@k for 1-k

    Arguments:
    ----------
    X: csr_matrix, np.ndarray or dict
        * csr_matrix: csr_matrix with nnz at relevant places
        * np.ndarray (float): scores for each label
            User must ensure shape is fine
        * np.ndarray (int): top indices (in sorted order)
            User must ensure shape is fine
        * {'indices': np.ndarray, 'scores': np.ndarray}
    true_labels: csr_matrix or np.ndarray
        ground truth in sparse or dense format
    inv_psp: np.ndarray
        propensity scores for each label
    k: int, optional (default=5)
        compute propensity scored nDCG till k
    sorted: boolean, optional, default=False
        whether X is already sorted (will skip sorting)
        * used when X is of type dict or np.ndarray (of indices)
        * shape is not checked is X are np.ndarray
        * must be set to true when X are np.ndarray (of indices)


    Returns:
    -------
    np.ndarray: propensity scored nDCG values for 1-k
    """
    indices, true_labels, ps_indices, inv_psp = _setup_metric(
        X, true_labels, inv_psp, k=k, sort_values=sort_values
    )
    eval_flags = _eval_flags(indices, true_labels, inv_psp)
    ps_eval_flags = _eval_flags(ps_indices, true_labels, inv_psp)
    _total_pos = np.asarray(true_labels.sum(axis=1), dtype=np.int32)
    _max_pos = max(np.max(_total_pos), k)
    _cumsum = np.cumsum(1 / np.log2(np.arange(1, _max_pos + 1) + 1))
    n = _cumsum[_total_pos - 1]
    return _ndcg(eval_flags, n, k) / _ndcg(ps_eval_flags, n, k)


def _ndcg(eval_flags, n, k=5):
    _cumsum = 0
    _dcg = np.cumsum(np.multiply(eval_flags, 1 / np.log2(np.arange(k) + 2)), axis=-1)
    ndcg = np.zeros((1, k), dtype=np.float32)
    for _k in range(k):
        _cumsum += 1 / np.log2(_k + 1 + 1)
        ndcg[0, _k] = np.mean(
            np.multiply(_dcg[:, _k].reshape(-1, 1), 1 / np.minimum(n, _cumsum))
        )
    return np.ravel(ndcg)


def recall(X, true_labels, k=5, sort_values=False):
    """
    Compute recall@k for 1-k

    Arguments:
    ----------
    X: csr_matrix, np.ndarray or dict
        * csr_matrix: csr_matrix with nnz at relevant places
        * np.ndarray (float): scores for each label
            User must ensure shape is fine
        * np.ndarray (int): top indices (in sorted order)
            User must ensure shape is fine
        * {'indices': np.ndarray, 'scores': np.ndarray}
    true_labels: csr_matrix or np.ndarray
        ground truth in sparse or dense format
    k: int, optional (default=5)
        compute recall till k
    sorted: boolean, optional, default=False
        whether X is already sorted (will skip sorting)
        * used when X is of type dict or np.ndarray (of indices)
        * shape is not checked is X are np.ndarray
        * must be set to true when X are np.ndarray (of indices)


    Returns:
    -------
    np.ndarray: recall values for 1-k
    """
    indices, true_labels, _, _ = _setup_metric(
        X, true_labels, k=k, sort_values=sort_values
    )
    deno = true_labels.sum(axis=1)
    deno[deno == 0] = 1
    deno = 1 / deno
    eval_flags = _eval_flags(indices, true_labels, None)
    return _recall(eval_flags, deno, k)


def hits(X, true_labels, k=5, sort_values=False):
    """
    Compute hits@k for 1-k

    Arguments:
    ----------
    X: csr_matrix, np.ndarray or dict
        * csr_matrix: csr_matrix with nnz at relevant places
        * np.ndarray (float): scores for each label
            User must ensure shape is fine
        * np.ndarray (int): top indices (in sorted order)
            User must ensure shape is fine
        * {'indices': np.ndarray, 'scores': np.ndarray}
    true_labels: csr_matrix or np.ndarray
        ground truth in sparse or dense format
    k: int, optional (default=5)
        compute recall till k
    sorted: boolean, optional, default=False
        whether X is already sorted (will skip sorting)
        * used when X is of type dict or np.ndarray (of indices)
        * shape is not checked is X are np.ndarray
        * must be set to true when X are np.ndarray (of indices)


    Returns:
    -------
    np.ndarray: hits values for 1-k
    """
    indices, true_labels, _, _ = _setup_metric(
        X, true_labels, k=k, sort_values=sort_values
    )
    eval_flags = _eval_flags(indices, true_labels, None)
    return _hits(eval_flags)


def _hits(eval_flags):
    eval_flags = np.clip(np.cumsum(eval_flags, axis=-1), 0, 1)
    hits = np.mean(eval_flags, axis=0)
    return np.ravel(hits)


def psrecall(X, true_labels, inv_psp, k=5, sort_values=False):
    """
    Compute propensity scored recall@k for 1-k

    Arguments:
    ----------
    X: csr_matrix, np.ndarray or dict
        * csr_matrix: csr_matrix with nnz at relevant places
        * np.ndarray (float): scores for each label
            User must ensure shape is fine
        * np.ndarray (int): top indices (in sorted order)
            User must ensure shape is fine
        * {'indices': np.ndarray, 'scores': np.ndarray}
    true_labels: csr_matrix or np.ndarray
        ground truth in sparse or dense format
    inv_psp: np.ndarray
        propensity scores for each label
    k: int, optional (default=5)
        compute propensity scored recall till k
    sorted: boolean, optional, default=False
        whether X is already sorted (will skip sorting)
        * used when X is of type dict or np.ndarray (of indices)
        * shape is not checked is X are np.ndarray
        * must be set to true when X are np.ndarray (of indices)


    Returns:
    -------
    np.ndarray: propensity scored recall values for 1-k
    """
    indices, true_labels, ps_indices, inv_psp = _setup_metric(
        X, true_labels, inv_psp, k=k, sort_values=sort_values
    )
    deno = true_labels.sum(axis=1)
    deno[deno == 0] = 1
    deno = 1 / deno
    eval_flags = _eval_flags(indices, true_labels, inv_psp)
    ps_eval_flags = _eval_flags(ps_indices, true_labels, inv_psp)
    return _recall(eval_flags, deno, k) / _recall(ps_eval_flags, deno, k)


def _recall(eval_flags, deno, k=5):
    eval_flags = np.cumsum(eval_flags, axis=-1)
    recall = np.mean(np.multiply(eval_flags, deno), axis=0)
    return np.ravel(recall)


def _auc(X, k):
    non_inv = np.cumsum(X, axis=1)
    cum_noninv = np.sum(np.multiply(non_inv, 1 - X), axis=1)
    n_pos = non_inv[:, -1]
    all_pairs = np.multiply(n_pos, k - n_pos)
    all_pairs[all_pairs == 0] = 1.0  # for safe divide
    point_auc = np.divide(cum_noninv, all_pairs)
    return np.mean(point_auc)


def auc(X, true_labels, k, sort_values=False):
    """
    Compute AUC score

    Arguments:
    ----------
    X: csr_matrix, np.ndarray or dict
        * csr_matrix: csr_matrix with nnz at relevant places
        * np.ndarray (float): scores for each label
            User must ensure shape is fine
        * np.ndarray (int): top indices (in sorted order)
            User must ensure shape is fine
        * {'indices': np.ndarray, 'scores': np.ndarray}
    true_labels: csr_matrix or np.ndarray
        ground truth in sparse or dense format
    k: int, optional (default=5)
        retain top-k predictions only
    sorted: boolean, optional, default=False
        whether X is already sorted (will skip sorting)
        * used when X is of type dict or np.ndarray (of indices)
        * shape is not checked is X are np.ndarray
        * must be set to true when X are np.ndarray (of indices)


    Returns:
    -------
    np.ndarray: auc score
    """
    indices, true_labels, _, _ = _setup_metric(
        X, true_labels, k=k, sort_values=sort_values
    )
    eval_flags = _eval_flags(indices, true_labels, None)
    return _auc(eval_flags, k)
