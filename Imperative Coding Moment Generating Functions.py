import numpy as np
import functools
from functools import reduce
from scipy.stats import kurtosis, skew
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

#Define Test Data

# Python variables can be capital letters
np.random.seed(5)
X = np.random.normal(size=100)
sns.set(color_codes=True)
sns.distplot(X)
print(X)

#Define a function first_moment that computes the MEAN

def first_moment(X):
    n = len(X)
    if n == 0:
        return 0
    sum = 0.0
    for i in range(len(X)):
        sum = sum + X[i]
    return float(sum/n)      

l = [1,2]
mu = first_moment(l)
print(mu)

#Define a function second_moment that computes the Standard Deviation

import numpy as np

def second_moment(X, mu):
    n = len(X)
    if n == 0:
        return 0
    sum = 0.0
    for i in range(len(X)):
        sum = sum + (X[i] - mu)**2
    return np.sqrt(float(sum/n))      

l = [1 , 2]
mu = first_moment(l)
sigma = second_moment(l, mu)
print(sigma)

#Define a function, third_moment, that computes the Skew

def third_moment(X, mu, sigma):
    n = len(X)
    if n == 0:
        return 0
    sum = 0.0
    for i in range(len(X)):
        sum = sum + (X[i] - mu)**3
    return float(float(sum/sigma**3)/n) 

l = [1 , 2]
mu = first_moment(l)
sigma = second_moment(l, mu)
gamma_1 = third_moment(l, mu, sigma)
print(gamma_1)

#Define a function, fourth_moment, to compute the Kurtosis

def fourth_moment(X, mu, sigma):
    n = len(X)
    if n == 0:
        return 0
    sum = 0.0
    for i in range(len(X)):
        sum = sum + (X[i] - mu)**4
    return float(float(sum/sigma**4)/n) - 3

l = [1 , 2]
gamma_2 = third_moment(l, mu, sigma)
print(gamma_2)

