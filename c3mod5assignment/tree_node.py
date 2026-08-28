'''
TreeNode in a DecisionTree classifier
The TreeNode is defined by a subset (list) of indices of observations
Expected:
- X NxD matrix (features)
- Y Nx1 matrix (target class known value from training data)
'''
import numpy as np

class TreeNode:
    def __init__(self, X, Y):
        self.i = []

        #We enforce that we have read only access to X and Y from this class
        self.X = np.asarray(X).view()
        self.Y = np.asarray(Y).view()
        self.X.flags.writeable = False
        self.Y.flags.writeable = False
        self.majority_class = None
        self.majority_probability = None
        self.correctcount = None
        self.errorcount = None

    def calculate_majority_class(self):
        self.majority_class = None
        self.majority_probability = None
        self.correctcount = 0
        self.errorcount = 0
        if len(self.i) == 0:
            return

        values, counts = np.unique(
            self.Y[self.i, 0],
            return_counts=True
        )

        majority_index = np.argmax(counts)
        self.majority_class = values[majority_index]
        self.correctcount = int(counts[majority_index])
        self.errorcount = len(self.i) - self.correctcount
        self.majority_probability = self.correctcount / len(self.i)

    def node_as_str(self):
        s = f"majority_class {self.majority_class}"
        s += f"\nmajority_probability {self.majority_probability}"
        s += f"\ncorrectcount {self.correctcount}"
        s += f"\nerrorcount {self.errorcount}"
        return s
