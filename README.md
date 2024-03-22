# Scikit-xml
This python package provides a comprehensive set of functions to compute advanced evaluation metrics for machine learning models,
especially focusing on scenarios where the standard metrics do not suffice. 

It includes specialized metrics such as precision@k, recall@k, f1@k, and normalized cumulative discount gain (NDCG) at k, along with propensity-scored versions of these metrics to handle cases with biased datasets.

## Features
- Precision@k: Computes the precision of the predictions at the top-k ranking positions.
- Recall@k: Computes the recall of the predictions at the top-k ranking positions.
- F1@k: Computes the F1 score, which is the harmonic mean of precision and recall, at the top-k ranking positions.
- NDCG@k: Computes the normalized cumulative discount gain at the top-k ranking positions, a metric useful for evaluating rankings.

- Propensity-scored Metrics: For all the above metrics, versions that take into account propensity scores are available, useful for biased datasets.
- Validation Utilities: Functions to validate the shapes and types of the input arrays to ensure compatibility with the metrics functions.


## Installation
To install this package, clone the repository and install the dependencies listed in `requirements.txt`. 
Ensure you have Python 3.10 or newer.

```bash
git clone https://bitbucket.org/ipazia1/scikit-xml.git
cd scikit-xml
pip install -r requirements.txt
```

# Usage
Here is a quick example of how to use the `precision_at_k_score` function:

```python
import numpy as np
from skxml import precision_at_k_score

# Example ground truth and predictions
y_true = np.array([[1, 0, 1], [0, 1, 1]])
y_pred = np.array([[0.8, 0.2, 0.4], [0.1, 0.6, 0.8]])

# Compute precision at k=2
precision = precision_at_k_score(y_true, y_pred, k=2)
print(f"Precision@2: {precision}")
```

## Contributing
Contributions are welcome! Please submit a pull request or open an issue if you have suggestions for improvements or have identified bugs.

## License
This project is licensed under the MIT License - see the LICENSE file for details.