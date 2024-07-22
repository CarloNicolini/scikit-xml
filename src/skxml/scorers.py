from sklearn.metrics import make_scorer
from .metrics import precision_at_k

precision_at_k_scorer = make_scorer(precision_at_k, greater_is_better=True)
