'''
Module doing AdaBoost algorithm
'''
import sys
sys.path.append("../c3mod5assignment")
from tree_binary_classifier import TreeBinaryClassifier
import numpy as np
from datetime import datetime

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
            oneclf.max_depth = 1
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

    def print_with_tms(self, message):
        mytimestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"{mytimestamp}|{message}")

    def boost_iteration(self, X, Y, t):
        #The alpha is already also stored as property of each learner
        #So we'll often propagate alpha from AdaBoost to each of its individual learners
        #e.g. self.clf[t].set_alpha(new_alpha=self.alpha)
        self.print_with_tms(f"boost_iteration t {t}")
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

    def predict_iteration(self, X, t):
        y_pred = self.clf[t].predict(X=X)
        return y_pred[:,0]

    def predict(self, X, use_n_estimators=None):
        if use_n_estimators is None:
            use_n_estimators = self.n_estimators

        if use_n_estimators > self.n_estimators:
            raise ValueError("use_n_estimators should not exceed the number of estimators currently defined")

        N = X.shape[0]
        predictions_per_learner = np.zeros((N, use_n_estimators))
        for t in range(use_n_estimators):
            y_pred_val = self.predict_iteration(X=X, t=t)
            predictions_per_learner[:, t] = y_pred_val
        weights_per_learner_matrix = np.asarray(self.weight_per_learner[:use_n_estimators]).reshape(-1, 1) #Tx1 matrix
        total_per_datapoint = np.matmul(predictions_per_learner, weights_per_learner_matrix)
        outcome_per_datapoint = np.where(
            total_per_datapoint >= 0, 1, -1
        )
        return outcome_per_datapoint

    def classification_error(self, X, Y, use_n_estimators=None):
        '''
        Compute final classification error for the AdaBoost class on data X with true values Y
        '''
        if use_n_estimators is None:
            use_n_estimators = self.n_estimators

        N = X.shape[0]
        assert Y.shape[0] == N
        if N == 0:
            return None

        Y_pred = self.predict(X=X, use_n_estimators=use_n_estimators)
        Y_pred_flat = np.asarray(Y_pred).reshape(-1)
        Y_flat = np.asarray(Y).reshape(-1)

        errorcount = 0
        for i in range(N):
            if Y_pred_flat[i] != Y_flat[i]:
                errorcount += 1

        return errorcount / N
