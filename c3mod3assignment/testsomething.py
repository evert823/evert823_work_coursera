import numpy as np

def probabilities_from_score_matrix(score_matrix):
    return 1.0 / (1.0 + np.exp(-score_matrix))

testmatrix1 = np.array([[-100.0, -30.0, -10.0],
                       [-1.0, -0.1, -0.01],
                       [0.01, 0.1, 1.0],
                       [10.0, 30.0, 100.0]])
testmatrix2 = probabilities_from_score_matrix(testmatrix1)

testmatrix3 = np.array([1, 1, -1, 1])
testmatrix4 = (testmatrix3 == 1).astype(float)
print(testmatrix4.shape)
print(testmatrix4)
