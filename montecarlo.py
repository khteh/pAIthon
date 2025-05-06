import numpy as np
from numpy.random import Generator, PCG64DXSM
rng = Generator(PCG64DXSM())

# Monte Carlo method: simulating random data and estimating probabilities by counting occurrences. 
def generate(p1, size):
    # change this so that it generates 10000 random zeros and ones
    # where the probability of one is p1
    return rng.choice([0,1], p=[1-p1, p1], size=size)

def count_seq(seq):
    # pad a with 0 at both sides for edge cases when a starts or ends with 1
    d = np.diff(np.pad(seq, pad_width=1, mode='constant'))
    # subtract indices when value changes from 0 to 1 from indices where value changes from 1 to 0
    occurrences = np.flatnonzero(d == -1) - np.flatnonzero(d == 1)
    print(f"seq: {seq}, d: {d}, occurrences: {occurrences}")
    count = 0
    for i in occurrences:
        if i >= 5:
            count += 1 + (i - 5)
    return count

def main(p1):
    seq = generate(p1, 10)
    return count_seq(seq)

if __name__ == "__main__":
    print(main(2/3))
