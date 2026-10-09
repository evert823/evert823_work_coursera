from datetime import datetime
from scipy import sparse
import numpy as np
from sklearn.metrics import pairwise_distances

class KMeansCluster:
    def __init__(self):
        self.IsVerbose = False
        self.label_per_data_point = None

    def print_with_tms(self, message):
        mytimestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"{mytimestamp}|{message}")

    def init_centroids_simple(self, X, k, seed=None):
        '''
        Randomly pick k data points which will be the centroids going forward
        Simple means: not the k-means++ way of selecting the initial centroids
        '''
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

    def assign_data_points_to_centroids(self, X, centroids):
        '''
        For each data point:
        - determine distance to each centroid

        '''
        self.label_per_data_point = None
        pair_wise_distances = pairwise_distances(X, centroids)

        #Now for each datapoint we need the k for which the distance is smallest compared to other k
        self.label_per_data_point = np.argmin(a=pair_wise_distances, axis=1)
        if self.IsVerbose == True:
            print(f"type(self.label_per_data_point) {type(self.label_per_data_point)}")
            print(f"self.label_per_data_point.shape {self.label_per_data_point.shape}")

    def revise_centroids(self, X, k, centroids):
        '''
        For each centroid = cluster centre
        - find all data points assigned to this centroid
        - take the mean of the feature values
        - update coordinates of centroid to the computed means
        '''
        for label in range(k):
            mask = self.label_per_data_point == label
            cluster = X[mask]
            if len(cluster) == 0:
                cluster_avg = None
            else:
                cluster_avg = np.asarray(cluster.mean(axis=0)).ravel()
            if cluster_avg is not None:
                centroids[label] = cluster_avg

        return centroids

    def fit(self, X, k, seed=None):

        if not sparse.issparse(X):
            raise TypeError("X must be a SciPy sparse matrix")

        centroids =  self.init_centroids_simple(X=X, k=k, seed=seed)
        self.assign_data_points_to_centroids(X=X, centroids=centroids)
        return centroids
