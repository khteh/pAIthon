import numpy as np
import matplotlib.pyplot as plt

def F1Score(truths, predictions):
    tp = np.sum((predictions == 1) & (truths == 1))
    fp = np.sum((predictions == 1) & (truths == 0))
    fn = np.sum((predictions == 0) & (truths == 1))
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return 2 * (precision * recall) / (precision + recall)
