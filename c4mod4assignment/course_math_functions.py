import numpy as np
'''
Functions given by course wrapped in class
'''
class CourseMathFunctions:
    def __init__(self):
        self.IsVerbose = False

    def log_sum_exp(self, Z):
        """ Compute log(\sum_i exp(Z_i)) for some array Z."""
        return np.max(Z) + np.log(np.sum(np.exp(Z - np.max(Z))))

    def loglikelihood(self, data, weights, means, covs):
        """ Compute the loglikelihood of the data for a Gaussian mixture model with the given parameters. """
        num_clusters = len(means)
        num_dim = len(data[0])
        
        ll = 0
        for d in data:
            
            Z = np.zeros(num_clusters)
            for k in range(num_clusters):
                
                # Compute (x-mu)^T * Sigma^{-1} * (x-mu)
                delta = np.array(d) - means[k]
                exponent_term = np.dot(delta.T, np.dot(np.linalg.inv(covs[k]), delta))
                
                # Compute loglikelihood contribution for this data point and this cluster
                Z[k] += np.log(weights[k])
                Z[k] -= 1/2. * (num_dim * np.log(2*np.pi) + np.log(np.linalg.det(covs[k])) + exponent_term)
                
            # Increment loglikelihood contribution of this data point across all clusters
            ll += self.log_sum_exp(Z)
            
        return ll
