from functools import partial

from sklearn.metrics import make_scorer

from skxml._sk_metrics import f1_at_k, ndcg_at_k, precision_at_k, recall_at_k

precision_at_k_scorer = make_scorer(
    precision_at_k, greater_is_better=True, response_method="predict_proba"
)
recall_at_k_scorer = make_scorer(
    recall_at_k, greater_is_better=True, response_method="predict_proba"
)
ndcg_at_k_scorer = make_scorer(
    ndcg_at_k, greater_is_better=True, response_method="predict_proba"
)
psprecision_at_k_scorer = make_scorer(
    partial(precision_at_k, propensity_coeff=(0.5, 0.4)),
    greater_is_better=True,
    response_method="predict_proba",
)
psrecall_at_k_scorer = make_scorer(
    partial(recall_at_k, propensity_coeff=(0.5, 0.4)),
    greater_is_better=True,
    response_method="predict_proba",
)
psf1_at_k_scorer = make_scorer(
    partial(f1_at_k, propensity_coeff=(0.5, 0.4)),
    greater_is_better=True,
    response_method="predict_proba",
)
