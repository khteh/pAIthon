import numpy, tensorflow as tf
def positional_encoding(positions, d_model):
    """
    Precomputes a matrix with all the positional encodings.
    
    Arguments:
        positions (int): Maximum number of positions to be encoded 
        d_model (int): Encoding size 
    
    Returns:
        pos_encoding (tf.Tensor): A matrix of shape (1, position, d_model) with the positional encodings
    """
    position = numpy.arange(positions)[:, numpy.newaxis]
    k = numpy.arange(d_model)[numpy.newaxis, :]
    i = k // 2
    
    # initialize a matrix angle_rads of all the angles 
    angle_rates = 1 / numpy.power(10000, (2 * i) / numpy.float32(d_model))
    angle_rads = position * angle_rates

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = numpy.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = numpy.cos(angle_rads[:, 1::2])
    
    pos_encoding = angle_rads[numpy.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)
