'''
Module doing AdaBoost algorithm
'''
import sys
sys.path.append("../c3mod5assignment")
from tree_binary_classifier import TreeBinaryClassifier
import numpy as np

class AdaBoost:
    def __init__(self):
        self.IsVerbose = False
        self.n_estimators = 0
        self.clf = []
        self.weighted_error_per_learner = []
        self.weight_per_learner = []

        self.alpha = None

    def set_n_estimators(self, n_estimators):
        self.n_estimators = n_estimators
        self.clf.clear()
        self.weighted_error_per_learner = [0.0 for tw in range(n_estimators)]
        self.weight_per_learner = [0.0 for tw in range(n_estimators)]
        for t in range(n_estimators):
            oneclf = TreeBinaryClassifier()
            oneclf.max_depth = 2
            oneclf.min_node_size = 0
            oneclf.error_reduction_threshold = -1.0
            self.clf.append(oneclf)

    def normalize_alpha(self):
        #normalize alpha to add up to total of 1.0
        total = np.sum(self.alpha)
        self.alpha = self.alpha / total

    def compute_weight_for_learner_from_its_error(self, w_e: float):
        if w_e <= 0.0:
            return 100.0
        if w_e >= 1.0:
            return -100.0
        #TODO build stopping condition
        a = (1 - w_e) / w_e
        return np.log(a) / 2

    def recompute_alpha(self, X, Y, t):
        y_pred = self.clf[t].predict(X=X)
        N = X.shape[0]
        for i in range(N):
            if y_pred[i, 0] == Y[i]:
                new_alpha_i = self.alpha[i] * np.exp(-1 * self.weight_per_learner[t])
            else:
                new_alpha_i = self.alpha[i] * np.exp(self.weight_per_learner[t])
            self.alpha[i] = new_alpha_i

    def boost_iteration(self, X, Y, t):
        #The alpha is already also stored as property of each learner
        #So we'll often propagate alpha from AdaBoost to each of its individual learners
        #e.g. self.clf[t].set_alpha(new_alpha=self.alpha)
        oneclf = self.clf[t]
        oneclf.set_alpha(new_alpha=self.alpha)
        oneclf.fit(X=X, Y=Y)
        self.weighted_error_per_learner[t] = oneclf.compute_weighted_error(X=X, Y=Y)
        self.weight_per_learner[t] = self.compute_weight_for_learner_from_its_error(w_e=self.weighted_error_per_learner[t])
        self.recompute_alpha(X=X, Y=Y, t=t)
        self.normalize_alpha()

    def fit(self, X, Y):
        if self.IsVerbose == True:
            print(f"X.shape {X.shape} Y.shape {Y.shape}")

        N = X.shape[0]
        self.alpha = np.ones(N, dtype=float)
        self.normalize_alpha()

        for t in range(len(self.clf)):
            self.boost_iteration(X=X, Y=Y, t=t)
