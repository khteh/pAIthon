from lifelines.utils import concordance_index

def cindex(y_true, scores):
    return concordance_index(y_true, scores)
