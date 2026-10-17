import numpy as np
from scipy.stats import multivariate_normal

'''
Student code for EM algorithm
'''
class EMModel:
    def __init__(self):
        self.IsVerbose = False

    def log_sum_exp(self, Z):
        """
        Given by course
        Compute log(\sum_i exp(Z_i)) for some array Z.
        """
        return np.max(Z) + np.log(np.sum(np.exp(Z - np.max(Z))))

    def loglikelihood(self, data, weights, means, covs):
        """
        Given by course
        Compute the loglikelihood of the data for a Gaussian mixture model with the given parameters.
        """
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

    def compute_covariances(self, data, resp, soft_counts, means):
        num_clusters = len(soft_counts)
        #D = len(data[0])
        D = data.shape[1]
        num_data = len(data)
        covariances = [np.zeros((D,D))] * num_clusters
        
        for k in range(num_clusters):
            weighted_sum = np.zeros((D, D), dtype=float)
            for i in range(num_data):
                v = data[i] - means[k]
                term = np.outer(v, v.T)
                weighted_sum += term * resp[i,k]
            if soft_counts[k] == 0.0:
                raise ValueError("soft_count for a cluster should not be zero")
            covariances[k] = weighted_sum / soft_counts[k]

        return covariances


    def EM(self, data, init_means, init_covariances, init_weights, maxiter=1000, thresh=1e-4):
        data = np.asarray(data, dtype=float)

        means = init_means[:]
        covariances = init_covariances[:]
        weights = init_weights[:]
        
        # Infer dimensions of dataset and the number of clusters
        num_data = len(data)
        #D = len(data[0])
        D = data.shape[1]
        num_clusters = len(means)
        
        # Initialize some useful variables
        resp = np.zeros((num_data, num_clusters))
        ll = self.loglikelihood(data=data,
                                weights=weights,
                                means=means,
                                covs=covariances)
        ll_trace = [ll]
        
        for it in range(maxiter):
            if it % 5 == 0:
                print("Iteration %s" % it)
            
            # E-step: compute responsibilities
            resp = self.compute_responsibilities(data=data,
                                                 weights=weights,
                                                 means=means,
                                                 covariances=covariances)

            # M-step
            # Compute the total responsibility assigned to each cluster, which will be useful when 
            # implementing M-steps below. In the lectures this is called N^{soft}
            soft_counts = self.compute_soft_counts(resp=resp)
            
            # Update the weight for cluster k using the M-step update rule for the cluster weight, \hat{\pi}_k.
            # YOUR CODE HERE
            weights = self.compute_weights(soft_counts=soft_counts)
            
            # Update means for cluster k using the M-step update rule for the mean variables.
            # This will assign the variable means[k] to be our estimate for \hat{\mu}_k.
            # YOUR CODE HERE
            means = self.compute_means(data=data,
                                       resp=resp,
                                       soft_counts=soft_counts)
            
            # Update covariances for cluster k using the M-step update rule for covariance variables.
            # This will assign the variable covariances[k] to be the estimate for \hat{\Sigma}_k.
            # YOUR CODE HERE
            covariances = self.compute_covariances(data=data,
                                                   resp=resp,
                                                   soft_counts=soft_counts,
                                                   means=means
                                                   )
            
            # Compute the loglikelihood at this iteration
            # YOUR CODE HERE
            ll_latest = self.loglikelihood(data=data,
                                    weights=weights,
                                    means=means,
                                    covs=covariances)

            ll_trace.append(ll_latest)
            
            # Check for convergence in log-likelihood and store
            if np.abs(ll_latest - ll) < thresh and ll_latest > -np.inf:
                break
            ll = ll_latest
        
        if it % 5 != 0:
            print("Iteration %s" % it)

        #GHCP suggestion: recompute resp before returning out
        out = {'weights': weights, 'means': means, 'covs': covariances, 'loglik': ll_trace, 'resp': resp}

        return out
