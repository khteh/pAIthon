from lifelines.utils import concordance_index

def CIndex(y_true, scores):
    return concordance_index(y_true, scores)
