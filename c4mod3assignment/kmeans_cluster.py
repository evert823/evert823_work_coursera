from datetime import datetime
from scipy import sparse
import numpy as np

class KMeansCluster:
    def __init__(self):
        pass

    def print_with_tms(self, message):
        mytimestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"{mytimestamp}|{message}")

    def init_centroids_simple(self, X, k, seed=None):
        '''
        Randomly pick k data points which will be the centroids going forward
        Simple means: not the k-means++ way of selecting the initial centroids
        '''
        if not sparse.issparse(X):
            raise TypeError("X must be a SciPy sparse matrix")

        N = X.shape[0]

        if seed is not None:
            np.random.seed(seed)

        #Pick K indices from range [0, N).
        rand_indices = np.random.randint(0, N, k)

        if len(np.unique(rand_indices)) != k:
            raise ValueError("rand_indices contains duplicate values")
        #rand_indices = np.random.choice(N, size=k, replace=False)

        centroids = X[rand_indices,:].toarray()

        return centroids

    def fit(self, X, k, seed=None):
        centroids =  self.init_centroids_simple(X=X, k=k, seed=seed)
        return centroids
