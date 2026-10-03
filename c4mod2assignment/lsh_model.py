from datetime import datetime
import numpy as np
from scipy import sparse
from itertools import combinations

class LSHModel:
    def __init__(self, h, L, random_seed=0):
        self.IsVerbose = False
        self.random_seed = random_seed
        self.h = h
        self.powers_of_two = (1 << np.arange(self.h - 1, -1, -1))
        self.r = 0
        self.L = L
        self.table = {}
        self.index_bits = None
        self.index_numbers = None
        self.searched_bins = []
        self.searched_d_i2 = []
        self.time_last_search_sec = None

        self.cached_searched_d_i2 = []
        self.cached_min_d_hd = []
        self.cached_best_i_hd = []

        self.include_identical = True

    def clear_lsh_cache(self):
        self.cached_searched_d_i2.clear()
        self.cached_min_d_hd.clear()
        self.cached_best_i_hd.clear()

    def print_with_tms(self, message):
        mytimestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"{mytimestamp}|{message}")

    def norm(self, x):
        sum_sq=x.dot(x.T)
        norm=np.sqrt(sum_sq)
        return(norm)

    def cosine_distance(self, x, y):
        xy = x.dot(y.T)
        dist = xy/(self.norm(x)*self.norm(y))
        result = 1-dist[0,0]
        if result < 0.0:
            result = 0.0
        return result

    def generate_random_vectors(self, dim):
        #These define the hyperplanes and a split into bins
        return np.random.randn(dim, self.h)

    def create_sample_table(self):
        #This is only for understanding handling of type and format of table
        self.table = {}
        self.table[0] = [10, 11, 12]
        self.table[1] = [13, 14, 15]

    def fit(self, X):

        self.clear_lsh_cache()

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
        self.index_numbers = np.matmul(self.index_bits, self.powers_of_two)
        if self.IsVerbose == True:
            print(f"self.powers_of_two.shape {self.powers_of_two.shape} self.index_numbers.shape {self.index_numbers.shape}")

        self.table = {}
        for i in range(N):
            idx = int(self.index_numbers[i])
            if idx not in self.table:
                self.table[idx] = []
            self.table[idx].append(i)

    def search_by_diff_pattern(self, X, i, idx_bit_i, diff):
        '''
        Here we search a specific neighbour bin that we found using diff-pattern diff
        '''
        min_d = -1
        best_i = -1
        idx_bit_diff = np.array(
            [idx_bit_i[j] if j not in diff else 1 - idx_bit_i[j]
             for j in range(self.h)],
            dtype=np.int8
        )
        idx_number_diff = int(idx_bit_diff.dot(self.powers_of_two))
        if idx_number_diff in self.table:
            self.searched_bins.append(idx_number_diff)
            for i2 in self.table[idx_number_diff]:
                if i2 != i or self.include_identical == True:
                    x = X[i,:]
                    y = X[i2,:]
                    d = self.cosine_distance(x=x, y=y)
                    self.searched_d_i2.append((d, i2))
                    if min_d < 0 or d < min_d:
                        min_d = d
                        best_i = i2
        return idx_bit_diff, idx_number_diff, min_d, best_i

    def search_exact_hamming_distance(self, X, i, idx_bit_i, hd):
        '''
        Here we search all neighbour bins that have a specific hamming distance hd to the query point
        '''
        if len(self.cached_best_i_hd) > hd:
            self.searched_d_i2.clear()
            self.searched_d_i2 = self.cached_searched_d_i2[hd].copy()
            min_d_hd = self.cached_min_d_hd[hd]
            best_i_hd = self.cached_best_i_hd[hd]
            return min_d_hd, best_i_hd

        min_d_hd = -1
        best_i_hd = -1
        for diff in combinations(range(self.h), hd):
            idx_bit_diff, idx_number_diff, min_d, best_i = self.search_by_diff_pattern(X=X,
                                                                i=i,
                                                                idx_bit_i=idx_bit_i,
                                                                diff=diff)
            if min_d_hd < 0 or (min_d > -1 and min_d < min_d_hd):
                min_d_hd = min_d
                best_i_hd = best_i

        '''
        We cache self.searched_d_i2, min_d_hd, best_i_hd
        For the same search X, i
        Per hd
        '''
        self.cached_searched_d_i2.append(self.searched_d_i2.copy())
        self.cached_min_d_hd.append(min_d_hd)
        self.cached_best_i_hd.append(best_i_hd)

        return min_d_hd, best_i_hd

    def report_searched_d_i2(self, k=10):
        '''
        Return [(i2, d), ...], sorted by distance descending.
        '''
        result = [
            (int(i2), float(d))
            for d, i2 in sorted(
                self.searched_d_i2,
                key=lambda item: item[0],
                reverse=False
            )
        ]
        return result[:k]

    def search(self, X, i, r, reuse_cache=False):
        '''
        i is the index of a data point from the dataset (sparse matrix) X that was used earlier for fit
        (so we assume that we search from documents already in our input dataser)
        '''
        if reuse_cache == False:
            self.clear_lsh_cache()

        if not sparse.issparse(X):
            raise TypeError("X must be a SciPy sparse matrix")
        start_search_datetime = datetime.now()

        self.r = r
        self.searched_bins = []
        self.searched_d_i2 = []
        min_d_overall = -1
        best_i_overall = -1
        #First we get the bit representation and the integer representation
        idx_bit_i = self.index_bits[i]

        for hd in range(self.r + 1):
            min_d_hd, best_i_hd = self.search_exact_hamming_distance(X=X,
                                                                     i=i,
                                                                     idx_bit_i=idx_bit_i,
                                                                     hd=hd)

            if min_d_overall < 0 or (min_d_hd > -1 and min_d_hd < min_d_overall):
                min_d_overall = min_d_hd
                best_i_overall = best_i_hd

            self.print_with_tms(f"hd {hd} min_d_overall {min_d_overall} best_i_overall {best_i_overall}")

        end_search_datetime = datetime.now()
        time_last_search = end_search_datetime - start_search_datetime
        self.time_last_search_sec = time_last_search.total_seconds()
        return min_d_overall, best_i_overall

    def brute_force_search(self, X, i, k):
        '''
        Find the k nearest neighbours to data point with index i
        Return results sorted by distance asc
        Brute force means one big scan over entire X
        '''
        if not sparse.issparse(X):
            raise TypeError("X must be a SciPy sparse matrix")
        intm_result = []
        N = X.shape[0]
        x = X[i,:]
        for i2 in range(N):
            if i2 != i or self.include_identical == True:
                y = X[i2,:]
                d = self.cosine_distance(x=x, y=y)
                intm_result.append((d, i2))

        result = [
            (int(i2), float(d))
            for d, i2 in sorted(
                intm_result,
                key=lambda item: item[0],
                reverse=False
            )
        ]
        return result[:k]
