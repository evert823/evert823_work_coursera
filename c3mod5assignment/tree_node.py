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

    def calculate_node_values(self, Y):
        self.majority_class = None
        self.majority_probability = None
        self.correctcount = 0
        self.errorcount = 0
        if len(self.i) == 0:
            self.is_leaf = True
            return

        values, counts = np.unique(
            Y[self.i, 0],
            return_counts=True
        )

        majority_index = np.argmax(counts)
        self.majority_class = values[majority_index]
        self.correctcount = int(counts[majority_index])
        self.errorcount = len(self.i) - self.correctcount
        if self.errorcount == 0:
            self.is_leaf = True
        self.majority_probability = self.correctcount / len(self.i)

    def apply_stopping_conditions_1_2(self, max_depth, min_node_size):
        #Stopping condition 1 For the deepest leafs current_depth eq. max_depth and we split no futher
        if self.current_depth >= max_depth:
            self.is_leaf = True
        #Stopping condition 2 If <= min_node_size data points in this node then we split no futher
        if len(self.i) <= min_node_size:
            self.is_leaf = True

    def node_as_str(self):
        s = f"majority_class {self.majority_class}"
        s += f"\nmajority_probability {self.majority_probability}"
        s += f"\ncorrectcount {self.correctcount}"
        s += f"\nerrorcount {self.errorcount}"
        return s
