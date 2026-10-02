import numpy as np
from scipy import sparse

class LSHModel:
    def __init__(self, h, r, L, random_seed=0):
        self.IsVerbose = False
        self.random_seed = random_seed
        self.h = h
        self.r = r
        self.L = L
        self.table = {}
        self.index_bits = None
        self.index_numbers = None

    def norm(self, x):
        sum_sq=x.dot(x.T)
        norm=np.sqrt(sum_sq)
        return(norm)

    def cosine_distance(self, x, y):
        xy = x.dot(y.T)
        dist = xy/(self.norm(x)*self.norm(y))
        return 1-dist[0,0]

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

        np.random.seed(self.random_seed)

        N = X.shape[0]
        D = X.shape[1]
        vectorset = self.generate_random_vectors(dim=D)
        if self.IsVerbose == True:
            print(f"N {N} D {D} h {self.h} vectorset.shape {vectorset.shape}")
        self.index_bits = ((X.dot(vectorset)) >= 0).astype(np.int8)
        if self.IsVerbose == True:
            print(f"self.index_bits.shape {self.index_bits.shape}")
            print(self.index_bits)
        powers_of_two = (1 << np.arange(self.h - 1, -1, -1))
        self.index_numbers = np.matmul(self.index_bits, powers_of_two)
        if self.IsVerbose == True:
            print(f"powers_of_two.shape {powers_of_two.shape} self.index_numbers.shape {self.index_numbers.shape}")

        self.table = {}
        for i in range(N):
            idx = int(self.index_numbers[i])
            if idx not in self.table:
                self.table[idx] = []
            self.table[idx].append(i)
