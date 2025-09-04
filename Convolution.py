import numpy, math, tensorflow as tf
from math import floor
from pathlib import Path
from utils.GPU import InitializeGPU
import numpy.lib.recfunctions as reconcile
from numpy.random import Generator, PCG64DXSM
import matplotlib.pyplot as plt
rng = Generator(PCG64DXSM())

def Padding(data, pad: int):
    """
    Pad with zeros all images of the dataset X. The padding is applied to the height and width of an image, 
    as illustrated in Figure 1.
    
    Argument:
    X -- python numpy array of shape (m, n_H, n_W, n_C) representing a batch of m images
    pad -- integer, amount of padding around each image on vertical and horizontal dimensions
    
    Returns:
    X_pad -- padded image of shape (m, n_H + 2 * pad, n_W + 2 * pad, n_C)
    """
    if pad < 1:
        return
    print(f"\n=== {Padding.__name__} ===")
    print(f"X.shape: {data.shape}, pad: {pad}")
    for x in data:
        print(f"x.shape: {x.shape}")
    data_padded = numpy.empty((data.shape[0], data.shape[1]+2*pad, data.shape[2]+2*pad, data.shape[3]))
    print(f"data_padded.shape: {data_padded.shape}")
    for x in data_padded:
        print(f"x_pad.shape: {x.shape}")
    for i, image in enumerate(data):
        print(f"image: {image.shape}")
        data_padded[i] = numpy.pad(image, ((pad, pad), (pad,pad), (0,0)), mode='constant', constant_values=(0,0))
    print ("data.shape =\n", data.shape)
    print ("data_padded.shape =\n", data_padded.shape)
    print ("data[0,1] =\n", data[0, 1])
    print ("data[1,1] =\n", data[1, 1])
    print ("data_padded[0] =\n", data_padded[0])
    print ("data_padded[1,1] =\n", data_padded[1, 1])
    assert (data_padded[0,0] == 0).all()
    if pad > 1:
        assert (data_padded[0,1] == 0).all()
        print ("data_padded[2,2] =\n", data_padded[2, 2])
    fig, ax = plt.subplots(1, 2)
    ax[0].set_title('data')
    ax[0].imshow(data[0, :, :, 0])
    ax[1].set_title('Padded data')
    ax[1].imshow(data_padded[0, :, :, 0])
    #plt.show()
    return data_padded

def conv_single_step():
    """
    Apply one filter defined by parameters W on a single slice (a_slice_prev) of the output activation 
    of the previous layer.
    
    Arguments:
    a_slice_prev -- slice of input data of shape (f, f, n_C_prev)
    W -- Weight parameters contained in a window - matrix of shape (f, f, n_C_prev)
    b -- Bias parameters contained in a window - matrix of shape (1, 1, 1)
    
    Returns:
    Z -- a scalar value, the result of convolving the sliding window (W, b) on a slice x of the input data
    """
    print(f"\n=== {conv_single_step.__name__} ===")
    a_slice_prev = numpy.random.randn(4, 4, 3)
    W = numpy.random.randn(4, 4, 3)
    b = numpy.random.randn(1, 1, 1)
    #(≈ 3 lines of code)
    # Element-wise product between a_slice_prev and W. Do not add the bias yet.
    # s = None
    # Sum over all entries of the volume s.
    # Z = None
    # Add bias b to Z. Cast b to a float() so that Z results in a scalar value.
    # Z = None
    Z = a_slice_prev * W
    Z = numpy.sum(Z)
    Z += b[0,0,0]
    print("Z =", Z)
    assert (type(Z) == numpy.float64), "You must cast the output to numpy float 64"
    #assert numpy.isclose(Z, -6.999089450680221), "Wrong value" This needs numpy.random.seed(). Otherwise it will fail as every run will have a random value

# GRADED FUNCTION: conv_forward

def conv_forward(A_prev, W, b, hparams):
    """
    Implements the forward propagation for a convolution function
    
    Arguments:
    A_prev -- output activations of the previous layer, 
        numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
    W -- Weights, numpy array of shape (f, f, n_C_prev, n_C)
    b -- Biases, numpy array of shape (1, 1, 1, n_C)
    hparameters -- python dictionary containing "stride" and "pad"
        
    Returns:
    Z -- conv output, numpy array of shape (m, n_H, n_W, n_C)
    cache -- cache of values needed for the conv_backward() function
    """
    print(f"\n=== {conv_forward.__name__} ===")
    print(f"A_prev: {A_prev.shape}")
    print(f"W: {W.shape}")
    # Retrieve dimensions from A_prev's shape (≈1 line)  
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    
    # Retrieve dimensions from W's shape (≈1 line)
    (f, f, n_C_prev, n_C) = W.shape
    
    # Retrieve information from "hparameters" (≈2 lines)
    stride = hparams["stride"]
    pad = hparams["pad"]
    print(f"stride: {stride}, pad: {pad}, filter: {f}x{f}")
    
    # Compute the dimensions of the CONV output volume using the formula given above. 
    # Hint: use int() to apply the 'floor' operation. (≈2 lines)
    n_H = floor((n_H_prev - f + 2 * pad) / stride) + 1
    n_W = floor((n_W_prev - f + 2 * pad) / stride) + 1
    
    # Initialize the output volume Z with zeros. (≈1 line)
    Z = numpy.zeros((m,n_H,n_W,n_C))
    print(f"Z: {Z.shape}")
    # Create A_prev_pad by padding A_prev
    A_prev_pad = Padding(A_prev, pad)
    print(f"A_prev_pad.shape: {A_prev_pad.shape}")
    for i in range(A_prev_pad.shape[0]): # loop over the batch of training examples
        a_prev_pad = A_prev_pad[i]   # Select ith training example's padded activation
        #print(f"\n{i}: a_prev_pad.shape: {a_prev_pad.shape}")
        vstart = 0
        for h in range(n_H): # loop over vertical axis of the output volume
            # Find the vertical start and end of the current "slice" (≈2 lines)
            vend = vstart + f
            #print(f"\nvertical axis {h}: {vstart} {vend}")
            hstart = 0
            for w in range(n_W): # loop over horizontal axis of the output volume
                # Find the horizontal start and end of the current "slice" (≈2 lines)
                hend = hstart + f
                #print(f"\nhorizontal axis {w}: {hstart} {hend}")
                
                for c in range(n_C): # loop over channels (= #filters) of the output volume
                    # Use the corners to define the (3D) slice of a_prev_pad (See Hint above the cell). (≈1 line)
                    a_slice_prev = a_prev_pad[vstart:vend, hstart:hend, :]
                    #print(f"a_slice_prev.shape: {a_slice_prev.shape}")
                    # Convolve the (3D) slice with the correct filter W and bias b, to get back one output neuron. (≈3 line)
                    weights = W[:,:,:, c]
                    biases = b[:,:,:, c]
                    #print(f"weights: {weights.shape}, biases: {biases}")
                    z = a_slice_prev * weights
                    #print(f"element-wise multiplication: ({z.shape})")
                    z = numpy.sum(z)
                    #print(f"sum: {z.shape} {z}")
                    z += biases
                    #print(f"added biases: {z.shape} {z}")
                    Z[i, h, w, c] = z.item()
                hstart += stride
            vstart += stride
    # Save information in "cache" for the backprop
    cache = (A_prev, W, b, hparams)
    return Z, cache

if __name__ == "__main__":
    data = numpy.random.randn(4, 3, 3, 2)
    Padding(data, 3)
    conv_single_step()
    
    numpy.random.seed(1)
    A_prev = numpy.random.randn(2, 5, 7, 4)
    W = numpy.random.randn(3, 3, 4, 8)
    b = numpy.random.randn(1, 1, 1, 8)
    hparameters = {"pad" : 1,
                "stride": 2}

    Z, cache_conv = conv_forward(A_prev, W, b, hparameters)
    z_mean = numpy.mean(Z)
    z_0_2_1 = Z[0, 2, 1]
    cache_0_1_2_3 = cache_conv[0][1][2][3]
    print("Z's mean =\n", z_mean)
    print("Z[0,2,1] =\n", z_0_2_1)
    print("cache_conv[0][1][2][3] =\n", cache_0_1_2_3)        