import numpy as np

def probabilities_from_score_matrix(score_matrix):
    return 1.0 / (1.0 + np.exp(-score_matrix))

testmatrix = np.array([[-100.0, -30.0, -10.0],
                       [-1.0, -0.1, -0.01],
                       [0.01, 0.1, 1.0],
                       [10.0, 30.0, 100.0]])
testmatrix2 = probabilities_from_score_matrix(testmatrix)
print(testmatrix.shape)
print(testmatrix2.shape)
print(testmatrix)
print(testmatrix2)
