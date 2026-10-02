import numpy as np
from scipy import sparse

class LSHModel:
    def __init__(self, h, r, L):
        self.IsVerbose = False
        np.random.seed(0)
        self.h = h
        self.r = r
        self.L = L
        self.table = {}

    def norm(self, x):
        sum_sq=x.dot(x.T)
        norm=np.sqrt(sum_sq)
        return(norm)

    def generate_random_vectors(self, dim):
        #These define the hyperplanes and a split into bins
        return np.random.randn(dim, self.h)

    def create_sample_table(self):
        #This is only for understanding handling of type and format of table
        self.table = {}
        self.table[0] = [10, 11, 12]
        self.table[1] = [13, 14, 15]

    def fit(self, X):
        if not sparse.issparse(X):
            raise TypeError("X must be a SciPy sparse matrix")
        N = X.shape[0]
        D = X.shape[1]
        vectorset = self.generate_random_vectors(dim=D)
        if self.IsVerbose == True:
            print(f"N {N} D {D} h {self.h} vectorset.shape {vectorset.shape}")
        index_bits = ((X.dot(vectorset)) >= 0).astype(np.int8)
        if self.IsVerbose == True:
            print(f"index_bits.shape {index_bits.shape}")
            print(index_bits)
        powers_of_two = (1 << np.arange(self.h - 1, -1, -1))
        index_numbers = np.matmul(index_bits, powers_of_two)
        if self.IsVerbose == True:
            print(f"powers_of_two.shape {powers_of_two.shape} index_numbers.shape {index_numbers.shape}")

        self.table = {}
        for i in range(N):
            idx = int(index_numbers[i])
            if idx not in self.table:
                self.table[idx] = []
            self.table[idx].append(i)
