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


    def compute_soft_counts(self, resp):
        # (This was given by course but straightforward)
        # Compute the total responsibility assigned to each cluster, which will be useful when 
        # implementing M-steps below. In the lectures this is called N^{soft}
        soft_counts = np.sum(resp, axis=0)
        return soft_counts

    def compute_weights(self, soft_counts):
        num_clusters = len(soft_counts)
        total = np.sum(soft_counts)

        if total > 0.0:
            weights = soft_counts / total
        else:
            raise ValueError("The total soft count must be greater than zero.")

        return weights

    def compute_means(self, data, resp, soft_counts):
        num_clusters = len(soft_counts)
        num_data = len(data)
        #D = len(data[0])
        D = data.shape[1]
        means = [np.zeros(D)] * num_clusters
        
        for k in range(num_clusters):
            weighted_sum = np.zeros(D, dtype=float)
            for i in range(num_data):
                term = resp[i,k] * data[i,:]
                weighted_sum += term
            if soft_counts[k] == 0.0:
                raise ValueError("soft_count for a cluster should not be zero")
            means[k] = weighted_sum / soft_counts[k]
        return means
