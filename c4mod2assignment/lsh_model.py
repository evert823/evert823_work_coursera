import numpy as np

class LSHModel:
    def __init__(self, h, r, L):
        self.IsVerbose = False
        np.random.seed(0)
        self.h = h
        self.r = r
        self.L = L

    def norm(self, x):
        sum_sq=x.dot(x.T)
        norm=np.sqrt(sum_sq)
        return(norm)

    def generate_random_vectors(self, num_vector, dim):
        #These define the hyperplanes and a split into bins
        return np.random.randn(dim, num_vector)
