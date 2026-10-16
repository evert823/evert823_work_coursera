from datetime import datetime
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import copy
from PIL import Image
from io import BytesIO

from course_math_functions import CourseMathFunctions

'''
Course 4 module 4 programming assignment 1 is apparently meant to be carried out with scikit-learn
'''

def print_with_tms(message):
    mytimestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{mytimestamp}|{message}")

PRINTSTUFF = False

cmf = CourseMathFunctions()
if PRINTSTUFF == True:
    cmf.IsVerbose = True

print_with_tms("script started")

path = os.path.join("C:\\", "Users", "Evert Jan", "courseradatascience",
                       "course04", "module04", "data")
imagepath = os.path.join(path, "images")

