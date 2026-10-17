import numpy as np
from scipy.stats import multivariate_normal

'''
Student code for EM algorithm
'''
class EMModel:
    def __init__(self):
        self.IsVerbose = False

    def compute_responsibilities(self, data, weights, means, covariances):
        '''
        E-step: compute responsibilities, given the current parameters
        data, weights are numpy arrays
        means, covariances are arrays of numpy arrays
        '''
        num_data = len(data)
        num_clusters = len(means)
        resp = np.zeros((num_data,num_clusters), dtype=float)

        for i in range(num_data):
            weighted_pdf_values = np.zeros((1,num_clusters), dtype=float)
            for k in range(num_clusters):
                weighted_pdf_values[0,k] = multivariate_normal.pdf(x=data[i],
                                            mean=means[k],
                                            cov=covariances[k]) * weights[k]
            #Now ensure that for each data point the responsibilities sum up to 1
            denominator = np.sum(weighted_pdf_values[0,:])
            if denominator > 0:
                resp[i,:] = weighted_pdf_values[0,:] / denominator
            else:
                resp[i,:] = 1.0 / num_clusters
        return resp
