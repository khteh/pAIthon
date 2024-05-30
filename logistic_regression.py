import math
import numpy as np

x = np.array([4, 3, 0])
c1 = np.array([-.5, .1, .08])
c2 = np.array([-.2, .2, .31])
c3 = np.array([.5, -.1, 2.53])

def sigmoid(z):
    # add your implementation of the sigmoid function here
    # s(z)=1÷(1+exp(−z))
    print(1/(1+math.exp(-z)))

# calculate the output of the sigmoid for x with all three coefficients
result1 = x @ c1
result2 = x @ c2
result3 = x @ c3
sigmoid(result1)
sigmoid(result2)
sigmoid(result3)