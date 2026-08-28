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

    def calculate_majority_class(self):
        values, counts = np.unique(
            self.Y[self.i, 0],
            return_counts=True
        )

        majority_index = np.argmax(counts)
        majority_class = values[majority_index]
        majority_probability = counts[majority_index] / len(self.i)

        return majority_class, majority_probability

