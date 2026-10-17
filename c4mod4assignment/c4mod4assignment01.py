from datetime import datetime
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import copy
from PIL import Image
from io import BytesIO

from course_math_functions import CourseMathFunctions
from em_model import EMModel

'''
Course 4 module 4 programming assignment 1 is apparently meant to be carried out with scikit-learn
'''

def print_with_tms(message):
    mytimestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{mytimestamp}|{message}")

PRINTSTUFF = False

cmf = CourseMathFunctions()
model = EMModel()
if PRINTSTUFF == True:
    cmf.IsVerbose = True
    model.IsVerbose = True

print_with_tms("script started")

path = os.path.join("C:\\", "Users", "Evert Jan", "courseradatascience",
                       "course04", "module04", "data")
imagepath = os.path.join(path, "images")

resp = model.compute_responsibilities(data=np.array([[1.,2.],[-1.,-2.]]), weights=np.array([0.3, 0.7]),
                                means=[np.array([0.,0.]), np.array([1.,1.])],
                                covariances=[np.array([[1.5, 0.],[0.,2.5]]), np.array([[1.,1.],[1.,2.]])])

if resp.shape==(2,2) and np.allclose(resp, np.array([[0.10512733, 0.89487267], [0.46468164, 0.53531836]])):
    print('Checkpoint passed!')
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
    print('Checkpoint passed!')
else:
    raise Exception('Check your code again.')

data_tmp = np.array([[1.,2.],[-1.,-2.]])
resp = model.compute_responsibilities(data=data_tmp, weights=np.array([0.3, 0.7]),
                                means=[np.array([0.,0.]), np.array([1.,1.])],
                                covariances=[np.array([[1.5, 0.],[0.,2.5]]), np.array([[1.,1.],[1.,2.]])])
counts = model.compute_soft_counts(resp)
means = model.compute_means(data_tmp, resp, counts)

if np.allclose(means, np.array([[-0.6310085, -1.262017], [0.25140299, 0.50280599]])):
    print('Checkpoint passed!')
else:
    raise Exception('Check your code again.')
