import numpy as np
import matplotlib.pyplot as plt

def F1Score(truths, predictions):
    """
    Skewed Datasets:
    Precision = #True positives / (#True + #False positives) = #True positives / (#Total predicted positives)
    Recall = #True positives / (#True positives + #False negatives) = #True positives / (#Total actual positives) - Helps detect if the learning algorithm is predicting negatives all the time because the value will be zero.
    0 Precision and/or Recall is bad.
    For use cases with skewed classes or a rare class, decently high precision and recall values helps reassures the usefulness of the learning algorithm.

    Trade-off between Precision and Recall:
    Logistic regression: 0 <= F(w,b) <= 1
    Predict 1 if F(w,b) >= threshold
    Predict 0 if F(w,b) < threshold
    To predict 1 only if very confident, use high value of threshold. This results in high precision, low recall
    To predict 1 even when in doubt, use low value of threshold. This results in low precision, high recall

    Picking the threshold is not something can be done with cross-validation. It's up to the business needs / use-case.
    To automatically decide on the best learning algorithm without having to manually select between Precision and Recall, choose the algorithm with the highest F1 score.
    F1 score = 2 * PR / (P + R) <= Harmonic mean. A mean calculation which pays attention to the lower value.

    Reference: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
    """
    #print(f"\n=== {F1Score.__name__} ===")
    #print(f"truths: {truths.shape}, predictions: {predictions.shape}")
    assert truths.shape == predictions.shape
    tp = np.sum((predictions == 1) & (truths == 1))
    fp = np.sum((predictions == 1) & (truths == 0))
    fn = np.sum((predictions == 0) & (truths == 1))
    #print(f"tp: {tp}, fp: {fp}, tp + fp: {tp + fp}")
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
