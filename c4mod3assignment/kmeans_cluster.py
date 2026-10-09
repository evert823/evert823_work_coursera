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
        #following syntax suggested by course
        #not because it's good but because because we need course compatibility

        centroids = X[rand_indices,:].toarray()

        return centroids

    def assign_data_points_to_centroids(self, X, centroids):
        '''
        For each data point:
        - determine distance to each centroid

        '''
        N = X.shape[0]
        pair_wise_distances = pairwise_distances(X, centroids, metric='euclidean')

        #Now for each datapoint we need the k for which the distance is smallest compared to other k
        label_per_data_point_new = np.argmin(a=pair_wise_distances, axis=1)

        if self.label_per_data_point is None:
            reassigned_count = N
        else:
            reassigned_count = np.count_nonzero(
                self.label_per_data_point != label_per_data_point_new
            )
        self.label_per_data_point = label_per_data_point_new

        if self.IsVerbose == True:
            print(f"type(self.label_per_data_point) {type(self.label_per_data_point)}")
            print(f"self.label_per_data_point.shape {self.label_per_data_point.shape}")

        return reassigned_count

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
            if cluster.shape[0] > 0:
                cluster_avg = np.asarray(
                    cluster.mean(axis=0)
                    ).ravel()
                centroids[label] = cluster_avg

        return centroids

    def compute_heterogeneity(self, X, k, centroids):
        '''
        Compute the sum of squared distances from each data point to its own assigned centroid
        '''
        heterogeneity = 0.0

        for label in range(k):
            mask = self.label_per_data_point == label
            cluster = X[mask]
            pair_wise_distances = pairwise_distances(
                cluster,
                centroids[label].reshape(1, -1),
                metric="euclidean"
            )
            squared_distances = pair_wise_distances ** 2
            heterogeneity += np.sum(squared_distances)

        return heterogeneity


    def stopcondition(self,
                      reassigned_count,
                      iternr,
                      max_iterations,
                      heterogeneity,
                      heterogeneity_prev,
                      epsilon):
        if reassigned_count == 0:
            return True
        if np.abs(heterogeneity - heterogeneity_prev) <= epsilon:
            return True
        if iternr + 1 >= max_iterations:
            return True
        return False

    def fit(self, X, k, seed=None, epsilon=100.0, max_iterations=5):
        print(f"\nk {k} seed {seed} epsilon {epsilon} max_iterations {max_iterations}\n")
        if not sparse.issparse(X):
            raise TypeError("X must be a SciPy sparse matrix")

        self.label_per_data_point = None
        centroids =  self.init_centroids_simple(X=X, k=k, seed=seed)

        heterogeneity = -1.0
        heterogeneity_prev = -2.0
        stop = False
        iternr = 0
        while stop == False:
            reassigned_count = self.assign_data_points_to_centroids(X=X, centroids=centroids)
            centroids = self.revise_centroids(X=X, k=k, centroids=centroids)
            heterogeneity_prev = heterogeneity
            heterogeneity = self.compute_heterogeneity(X=X, k=k, centroids=centroids)
            print(f"iternr {iternr} reassigned_count {reassigned_count} heterogeneity {heterogeneity}")
            stop = self.stopcondition(reassigned_count=reassigned_count,
                                      iternr=iternr,
                                      max_iterations=max_iterations,
                                      heterogeneity=heterogeneity,
                                      heterogeneity_prev=heterogeneity_prev,
                                      epsilon=epsilon)
            iternr += 1

        return centroids
