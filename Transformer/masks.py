import tensorflow as tf
def create_look_ahead_mask(sequence_length):
    """
    Returns a lower triangular matrix filled with ones. This helps softmax computation give the appropriate weights to the words in the input sequence.
    
    Arguments:
        sequence_length (int): matrix size
    
    Returns:
        mask (tf.Tensor): binary tensor of size (sequence_length, sequence_length)
    """
    return tf.linalg.band_part(tf.ones((1, sequence_length, sequence_length)), -1, 0)

def create_padding_mask(decoder_token_ids):
    """
    Creates a matrix mask for the padding cells. This helps softmax computation give the appropriate weights to the words in the input sequence.
    
    Arguments:
        decoder_token_ids (matrix like): matrix of size (n, m)
    
    Returns:
        mask (tf.Tensor): binary tensor of size (n, 1, m)
    """    
    seq = 1 - tf.cast(tf.math.equal(decoder_token_ids, 0), tf.float32)

    # add extra dimensions to add the padding to the attention logits. 
    # this will allow for broadcasting later when comparing sequences
    return seq[:, tf.newaxis, :] 
