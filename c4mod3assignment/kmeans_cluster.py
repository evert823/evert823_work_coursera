from datetime import datetime
from scipy import sparse
import numpy as np
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt

class KMeansCluster:
    def __init__(self):
        self.IsVerbose = False
        self.label_per_data_point = None

    def print_with_tms(self, message):
        mytimestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"{mytimestamp}|{message}")

    def pick_proportional(self, weights):
        probabilities = weights / np.sum(weights)
        i = int(np.random.choice(
            weights.shape[0],
            p=probabilities
        ))
        return i

    def init_centroids_kpp(self, X, k, seed=None):
        '''
        Initialize the centroids using the k-means++ method
        '''
        N = X.shape[0]
        D = X.shape[1]

        if seed is not None:
            np.random.seed(seed)

        rand_indices = np.zeros(k, dtype=int)
        centroids = np.zeros((k, D), dtype=float)

        #Choose the first centroid uniformly random
        rand_indices[0] = np.random.randint(0, N)
        centroids[0, :] = X[rand_indices[0], :].toarray().ravel()

        k_chosen = 1
        while k_chosen < k:
            #Compute pair_wise_distances all data points all chosen centroids
            pair_wise_distances = pairwise_distances(
                X,
                centroids[:k_chosen, :],
                metric="euclidean"
            )

            #For each data point the distance to the nearest centroid
            distances_nearest_centroids = np.min(a=pair_wise_distances, axis=1)

            #Pick next centroid from data points
            #with probability proportional to squared distance to nearest centroid
            squared_distances = (distances_nearest_centroids ** 2).ravel()
            rand_indices[k_chosen] = self.pick_proportional(weights=squared_distances)
            centroids[k_chosen, :] = X[rand_indices[k_chosen], :].toarray().ravel()
            k_chosen += 1

        if len(np.unique(rand_indices)) != k:
            raise ValueError("rand_indices contains duplicate values")

        return centroids


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

    def report_label_per_data_point(self):
        '''
        Get the number of data points assigned to each label,
        plus the largest cluster and its label.
        '''
        if self.label_per_data_point is None:
            return {}

        labels, counts = np.unique(
            self.label_per_data_point,
            return_counts=True
        )

        report = dict(zip(labels.tolist(), counts.tolist()))

        largest_index = np.argmax(counts)
        report["largest_cluster_label"] = int(labels[largest_index])
        report["largest_cluster_size"] = int(counts[largest_index])

        return report

    def fit(self, X, k, seed=None, epsilon=100.0, max_iterations=5, use_kpp_method=False):
        self.print_with_tms(f"\nk {k} seed {seed} epsilon {epsilon} max_iterations {max_iterations} use_kpp_method {use_kpp_method}\n")
        if not sparse.issparse(X):
            raise TypeError("X must be a SciPy sparse matrix")

        self.label_per_data_point = None

        if use_kpp_method == False:
            centroids =  self.init_centroids_simple(X=X, k=k, seed=seed)
        else:
            centroids =  self.init_centroids_kpp(X=X, k=k, seed=seed)

        heterogeneity = -1.0
        heterogeneity_prev = -2.0
        stop = False
        iternr = 0
        while stop == False:
            reassigned_count = self.assign_data_points_to_centroids(X=X, centroids=centroids)
            centroids = self.revise_centroids(X=X, k=k, centroids=centroids)
            heterogeneity_prev = heterogeneity
            heterogeneity = self.compute_heterogeneity(X=X, k=k, centroids=centroids)
            if iternr % 20 == 0:
                print(f"iternr {iternr} reassigned_count {reassigned_count} heterogeneity {heterogeneity}")
            stop = self.stopcondition(reassigned_count=reassigned_count,
                                      iternr=iternr,
                                      max_iterations=max_iterations,
                                      heterogeneity=heterogeneity,
                                      heterogeneity_prev=heterogeneity_prev,
                                      epsilon=epsilon)
            iternr += 1
        self.print_with_tms(f"\nk {k} seed {seed} epsilon {epsilon} true_iterations {iternr - 1}\n")

        return centroids, heterogeneity

    def fit_multiple_init_one_k(self, X, k,
                                seed_array=[0],
                                epsilon=100.0, max_iterations=5,
                                use_kpp_method=False):
        '''
        Input k is fixed
        For each seed in seed_array
        - retry self.fit - this resets the centroids
        - capture heterogeneity
        At the end capture
        - best found heterogeneity
        - set self.label_per_data_point to distribution that was found with that best heterogeneity
        '''
        best_heterogeneity = -1.0
        best_seed = -1
        best_centroids = None
        best_label_per_data_point = None
        for seed in seed_array:
            centroids, heterogeneity = self.fit(X=X, k=k, seed=seed,
                                                epsilon=epsilon, max_iterations=max_iterations,
                                                use_kpp_method=use_kpp_method)
            if best_heterogeneity == -1 or (heterogeneity >= 0.0 and best_heterogeneity > heterogeneity):
                best_heterogeneity = heterogeneity
                best_seed = seed
                best_label_per_data_point = self.label_per_data_point.copy()
                best_centroids = centroids.copy()

        self.label_per_data_point = best_label_per_data_point.copy()

        return best_centroids, best_heterogeneity, best_seed

    def k_search(self, X, k_array=[3], seed_array=[0],
                 epsilon=100.0, max_iterations=5,
                 use_kpp_method=False,
                 png_file_path="a.png",
                 log_file_path="a.log"):
        '''
        Rerun fit_multiple_init_one_k for several values of k
        Use a fixed seed_array for each of these reruns
        Capture the found heterogeneity for each k
        Plot heterogeneity (y-axis) against k (x-axis)
        '''
        heterogeneity_results = []
        for k in k_array:
            file2 = open(log_file_path, 'a')
            file2.write(f'k {k} started {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
            file2.close()
            _, heterogeneity, _ = self.fit_multiple_init_one_k(X=X, k=k,
                                                               seed_array=seed_array,
                                                               epsilon=epsilon, max_iterations=max_iterations,
                                                               use_kpp_method=use_kpp_method)
            heterogeneity_results.append(heterogeneity)
            file2 = open(log_file_path, 'a')
            file2.write(f'k {k} completed {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
            file2.close()

        plt.figure()
        plt.plot(k_array, heterogeneity_results, marker="o")
        plt.xlabel("k")
        plt.ylabel("Heterogeneity")
        plt.title("Heterogeneity versus k")
        plt.grid(True)
        plt.savefig(png_file_path)
        plt.close()

        return heterogeneity_results
