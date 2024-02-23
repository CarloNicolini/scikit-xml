import numpy as np

def precision_at_k(y_true: np.ndarray, y_score:np.ndarray, k:int =3):
    """
    Compute precision at k.

    Parameters:
        y_true (np.ndarray): True labels.
        y_score (np.ndarray): Predicted scores.
        k (int): Number of top elements to consider. Default is 3.

    Returns:
        precision (float): Precision at k.
    """
    if y_score.ndim == 1:
        y_score = y_score.reshape(-1, len(y_score))
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, len(y_true))
    sorted_scores = np.argsort(y_score, axis=1)[:, ::-1]
    top_k = np.take_along_axis(arr=y_true, indices=sorted_scores, axis=1)[:, :k]
    precision = np.mean(top_k)
    
    return precision

def dcg_at_k(y_true: np.ndarray, y_score:np.ndarray, k:int =3):
    """
    Compute Discounted Cumulative Gain (DCG) at k.

    Parameters:
        y_true (np.ndarray): True relevance labels.
        y_score (np.ndarray): Predicted scores.
        k (int): Number of top elements to consider. Default is 3.

    Returns:
        dcg (float): DCG at k.
    """
    if y_score.ndim == 1:
        y_score = y_score.reshape(-1, len(y_score))
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, len(y_true))
    sorted_scores = np.argsort(y_score, axis=1)[:, ::-1]
    top_k = np.take_along_axis(arr=y_true, indices=sorted_scores, axis=1)[:, :k]
    rel_gain = top_k / np.log2(np.arange(2, k + 2))
    dcg = np.sum(rel_gain)
    return dcg

def ndcg_at_k(y_true: np.ndarray, y_score:np.ndarray, k:int =3):
    """
    Compute Normalized Discounted Cumulative Gain (DCG) at k.

    Parameters:
        y_true (np.ndarray): True relevance labels.
        y_score (np.ndarray): Predicted scores.
        k (int): Number of top elements to consider. Default is 3.

    Returns:
        ndcg (float): nDCG at k.
    """
    if y_score.ndim == 1:
        y_score = y_score.reshape(-1, len(y_score))
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, len(y_true))

    dcg = dcg_at_k(y_true=y_true, y_score=y_score)
    rel_gain = 1 / np.log2(np.arange(2, k + 2))
    ndcg = dcg/np.sum(rel_gain)
    return ndcg

def propensity(y_true: np.ndarray, A:float =0.55, B:float =1.5):
    """
    Compute the propensity scores using the sigmoidal function.

    Parameters:
        y_true (np.ndarray): True labels.
        A (float): Specific Parameter. Default is 0.55.
        B (float): Specific Parameter. Default is 1.5.

    Returns:
        propensity_scores (np.array): Propensity scores computed using the sigmoidal function.
    """
    N = y_true.shape[0]  # Size of the dataset
    Nl = y_true.sum(axis=0)  # Sum along rows to get frequency of each label
    C = (np.log(N)- 1) * (B + 1)**A
    propensity_scores = 1 / (1 + C * np.exp(-A * np.log(Nl + B)))
    return propensity_scores

def ps_precision_at_k(y_true: np.ndarray, y_score:np.ndarray, k:int =3):
    """
    Compute propensity scored precision at k.

    Parameters:
        y_true (np.ndarray): True labels.
        y_score (np.ndarray): Predicted scores.
        k (int): Number of top elements to consider. Default is 3.

    Returns:
        ps_precision (float): propensity scored precision at k.
    """
    if y_score.ndim == 1:
        y_score = y_score.reshape(-1, len(y_score))
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, len(y_true))

    sorted_scores = np.argsort(y_score, axis=1)[:, ::-1]
    top_k_y_true = np.take_along_axis(arr=y_true, indices=sorted_scores, axis=1)[:, :k]
    
    propensity_scores = propensity(y_true)

    psp_at_k = np.mean(top_k_y_true/propensity_scores[top_k_y_true])
    
    return psp_at_k

def ps_dcg_at_k(y_true: np.ndarray, y_score:np.ndarray, k:int =3):
    """
    Compute propensity scored Discounted Cumulative Gain (DCG) at k.

    Parameters:
        y_true (np.ndarray): True relevance labels.
        y_score (np.ndarray): Predicted scores.
        k (int): Number of top elements to consider. Default is 3.

    Returns:
        ps_dcg (float): propensity scored DCG at k.
    """
    if y_score.ndim == 1:
        y_score = y_score.reshape(-1, len(y_score))
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, len(y_true))
        
    sorted_scores = np.argsort(y_score, axis=1)[:, ::-1]
    top_k_y_true = np.take_along_axis(arr=y_true, indices=sorted_scores, axis=1)[:, :k]

    propensity_scores = propensity(y_true)

    rel_gain = top_k_y_true /(propensity_scores[top_k_y_true]* np.log2(np.arange(2, k + 2)))
    ps_dcg_at_k = np.sum(rel_gain)

    return ps_dcg_at_k

def ps_ndcg_at_k(y_true: np.ndarray, y_score:np.ndarray, k:int =3):
    """
    Compute propensity scored Normalized Discounted Cumulative Gain (DCG) at k.

    Parameters:
        y_true (np.ndarray): True relevance labels.
        y_score (np.ndarray): Predicted scores.
        k (int): Number of top elements to consider. Default is 3.

    Returns:
        ps_ndcg (float): propensity scored nDCG at k.
    """
    if y_score.ndim == 1:
        y_score = y_score.reshape(-1, len(y_score))
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, len(y_true))

    ps_dcg = ps_dcg_at_k(y_true=y_true, y_score=y_score)

    rel_gain = 1 / np.log2(np.arange(2, k + 2))
    ps_ndcg_at_k = ps_dcg/np.sum(rel_gain)
    
    return ps_ndcg_at_k