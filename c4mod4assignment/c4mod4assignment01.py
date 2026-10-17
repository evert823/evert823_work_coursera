from datetime import datetime
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import copy
from PIL import Image
from io import BytesIO

from em_model import EMModel

'''
Course 4 module 4 programming assignment 1 is apparently meant to be carried out with scikit-learn
'''

def print_with_tms(message):
    mytimestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{mytimestamp}|{message}")


def generate_MoG_data(num_data, means, covariances, weights):
    """ Creates a list of data points """
    num_clusters = len(weights)
    data = []
    for i in range(num_data):
        #  Use np.random.choice and weights to pick a cluster id greater than or equal to 0 and less than num_clusters.
        k = np.random.choice(len(weights), 1, p=weights)[0]

        # Use np.random.multivariate_normal to create data from this cluster
        x = np.random.multivariate_normal(means[k], covariances[k])

        data.append(x)
    return data

def plot_to_png(data, output_dir, file_name):
    plt.figure()
    d = np.vstack(data)
    plt.plot(d[:,0], d[:,1],'ko')
    plt.rcParams.update({'font.size':16})
    plt.tight_layout()
    output_file = os.path.join(output_dir, file_name)
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

PRINTSTUFF = False

model = EMModel()
if PRINTSTUFF == True:
    model.IsVerbose = True

print_with_tms("script started")

path = os.path.join("C:\\", "Users", "Evert Jan", "courseradatascience",
                       "course04", "module04", "data")
imagepath = os.path.join(path, "images")

output_dir = os.path.join(os.path.dirname(__file__), "output")

resp = model.compute_responsibilities(data=np.array([[1.,2.],[-1.,-2.]]), weights=np.array([0.3, 0.7]),
                                means=[np.array([0.,0.]), np.array([1.,1.])],
                                covariances=[np.array([[1.5, 0.],[0.,2.5]]), np.array([[1.,1.],[1.,2.]])])

if resp.shape==(2,2) and np.allclose(resp, np.array([[0.10512733, 0.89487267], [0.46468164, 0.53531836]])):
    print('Checkpoint 1 passed!')
else:
    raise Exception('Check your code again.')

resp = model.compute_responsibilities(data=np.array([[1.,2.],[-1.,-2.],[0,0]]), weights=np.array([0.3, 0.7]),
                                means=[np.array([0.,0.]), np.array([1.,1.])],
                                covariances=[np.array([[1.5, 0.],[0.,2.5]]), np.array([[1.,1.],[1.,2.]])])
counts = model.compute_soft_counts(resp)
weights = model.compute_weights(counts)

print(counts)
print(weights)

if np.allclose(weights, [0.27904865942515705, 0.720951340574843]):
    print('Checkpoint 2 passed!')
else:
    raise Exception('Check your code again.')

data_tmp = np.array([[1.,2.],[-1.,-2.]])
resp = model.compute_responsibilities(data=data_tmp, weights=np.array([0.3, 0.7]),
                                means=[np.array([0.,0.]), np.array([1.,1.])],
                                covariances=[np.array([[1.5, 0.],[0.,2.5]]), np.array([[1.,1.],[1.,2.]])])
counts = model.compute_soft_counts(resp)
means = model.compute_means(data_tmp, resp, counts)

if np.allclose(means, np.array([[-0.6310085, -1.262017], [0.25140299, 0.50280599]])):
    print('Checkpoint 3 passed!')
else:
    raise Exception('Check your code again.')

data_tmp = np.array([[1.,2.],[-1.,-2.]])
resp = model.compute_responsibilities(data=data_tmp, weights=np.array([0.3, 0.7]),
                                means=[np.array([0.,0.]), np.array([1.,1.])],
                                covariances=[np.array([[1.5, 0.],[0.,2.5]]), np.array([[1.,1.],[1.,2.]])])
counts = model.compute_soft_counts(resp)
means = model.compute_means(data_tmp, resp, counts)
covariances = model.compute_covariances(data_tmp, resp, counts, means)

if np.allclose(covariances[0], np.array([[0.60182827, 1.20365655], [1.20365655, 2.4073131]])) and \
    np.allclose(covariances[1], np.array([[ 0.93679654, 1.87359307], [1.87359307, 3.74718614]])):
    print('Checkpoint 4 passed!')
else:
    raise Exception('Check your code again.')


#Below is creation of some testdata and testing the EM method of our model

# Model parameters
init_means = [
    [5, 0], # mean of cluster 1
    [1, 1], # mean of cluster 2
    [0, 5]  # mean of cluster 3
]
init_covariances = [
    [[.5, 0.], [0, .5]], # covariance of cluster 1
    [[.92, .38], [.38, .91]], # covariance of cluster 2
    [[.5, 0.], [0, .5]]  # covariance of cluster 3
]
init_weights = [1/4., 1/2., 1/4.]  # weights of each cluster

# Generate data
np.random.seed(4)
data = generate_MoG_data(100, init_means, init_covariances, init_weights)

plot_to_png(data=data, output_dir=output_dir, file_name="MoG_data")

np.random.seed(4)

# Initialization of parameters
chosen = np.random.choice(len(data), 3, replace=False)
initial_means = [data[x] for x in chosen]
initial_covs = [np.cov(data, rowvar=0)] * 3
initial_weights = [1/3.] * 3

# Run EM 
results = model.EM(data=data,
                   init_means=initial_means,
                   init_covariances=initial_covs,
                   init_weights=initial_weights)
print(results)
