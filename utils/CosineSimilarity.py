import numpy
def cosine_similarity(u: numpy.ndarray, v: numpy.ndarray) -> float:
    """Compute the cosine similarity between two vectors"""
    # Special case. Consider the case u = [0, 0], v=[0, 0]
    if numpy.all(u == v):
        return 1
    norm_u = numpy.linalg.norm(u, ord=2)
    norm_v = numpy.linalg.norm(v, ord=2)
    # Avoid division by 0
    return 0 if numpy.isclose(norm_u * norm_v, 0, atol=1e-32) else (u @ v) / (norm_u * norm_v)
