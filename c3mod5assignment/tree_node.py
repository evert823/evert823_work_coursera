'''
TreeNode in a DecisionTree classifier
The TreeNode is defined by a subset (list) of indices of observations
Expected:
- X NxD matrix (features)
- Y Nx1 matrix (target class known value from training data)
'''
import numpy as np

class TreeNode:
    def __init__(self):
        self.i = []

        #We enforce that we have read only access to X and Y from this class
        self.majority_class = None
        self.majority_probability = None
        self.correctcount = None
        self.errorcount = None
        self.best_j_from_node = -1
        self.is_leaf = False
        self.used_features_current_path = []
        self.current_depth = -1

    def get_values_weighted_counts(self, Y, alpha):
        values = np.unique(Y[self.i, 0])
        weighted_counts = np.zeros(values.shape)
        for dpi in self.i:
            for k in range(values.shape[0]):
                if Y[dpi, 0] == values[k]:
                    weighted_counts[k] += alpha[dpi]
        return values, weighted_counts


    def calculate_node_values(self, Y, alpha):
        self.majority_class = None
        self.majority_probability = None
        self.correctcount = 0
        self.errorcount = 0
        if len(self.i) == 0:
            self.is_leaf = True
            return

        values, weighted_counts_per_value = self.get_values_weighted_counts(Y=Y, alpha=alpha)

        majority_index = np.argmax(weighted_counts_per_value)
        self.majority_class = values[majority_index]
        self.correctcount = weighted_counts_per_value[majority_index]
        totalweight_current_node = np.sum(alpha[self.i])
        self.errorcount = totalweight_current_node - self.correctcount
        if self.errorcount == 0.0:
            self.is_leaf = True
        self.majority_probability = self.correctcount / totalweight_current_node

    def apply_stopping_conditions_1_2(self, max_depth, min_node_size):
        #Stopping condition 1 For the deepest leafs current_depth eq. max_depth and we split no futher
        if self.current_depth >= max_depth:
            self.is_leaf = True

        #Stopping condition 2 If <= min_node_size data points in this node then we split no futher
        #Explicit choice: here we do NOT use totalweight_current_node
        if len(self.i) <= min_node_size:
            self.is_leaf = True

    def node_as_str(self):
        s = f"majority_class {self.majority_class}"
        s += f"\nmajority_probability {self.majority_probability}"
        s += f"\ncorrectcount {self.correctcount}"
        s += f"\nerrorcount {self.errorcount}"
        return s
