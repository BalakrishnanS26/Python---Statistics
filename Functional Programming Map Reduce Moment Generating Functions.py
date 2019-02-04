# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 00:31:41 2019

@author: Balakrishnan
"""

import numpy as np
import functools
from functools import reduce
from scipy.stats import kurtosis, skew
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

#Define test data set X

np.random.seed(5)
X = np.random.normal(size=100)
sns.set(color_codes=True)
sns.distplot(X)
print(X)

#Define a function, first_moment_term, to compute the MEAN

def first_moment_term(x):
    return x

x = 1

new_term = first_moment_term(x)

print(new_term)

def my_mean(x):
    n = len(x)
    if n == 0 :
        return 0.0
    transformedX = map(lambda x : first_moment_term(x), x)
    sumX = reduce(lambda x, y : x + y, transformedX)
    return sumX/float(n)

test_list = [1,2,3] 
mu = my_mean(test_list)
print(mu)

#Define a function, second_moment_term, to compute the standard deviation

def second_moment_term(x, mu):
    term = (x - mu)**2
    return term

val = 5
mu = 3
term = second_moment_term(x, mu)

def my_stdev(test_list,mu):
    n = len(test_list)
    if n == 0 :
        return 0.0
    transformedX = map(lambda x : second_moment_term(x,mu), test_list)
    sumX = reduce(lambda x, y : x + y, transformedX)
    return np.sqrt(sumX/float(n))

test_list = [1,2,3] 
mu = my_mean(test_list)
print(mu)
sigma = my_stdev(test_list, mu)
print(sigma)

#Define a function, third_moment_term, to compute the Skew

def third_moment_term(x, mu, sigma):
    gamma_1 = (x - mu)**3/(sigma**3)
    return gamma_1

x = 5
mu = 3
sigma = 1
gamma_1 = third_moment_term(x, mu, sigma)

def my_skew(test_list,mu,sigma):
    n = len(test_list)
    if n == 0 :
        return 0.0
    transformedX = map(lambda x : third_moment_term(x,mu,sigma), test_list)
    sumX = reduce(lambda x, y : x + y, transformedX)
    return float(sumX)/float(n)

test_list = [1,2,3] 
mu = my_mean(test_list)
print(mu)
sigma = my_stdev(test_list, mu)
print(sigma)
gamma_1 = my_skew(test_list, mu, sigma)
print(gamma_1)

#Define a function, fourth_moment_term, to compute the kurtosis

def fourth_moment_term(x, mu, sigma):
    gamma = (x - mu)**4/(sigma**4)
    return gamma  

x = 5
mu = 3
sigma = 1
gamma = fourth_moment_term(x, mu, sigma)

def my_kurtosis(test_list,mu,sigma):
    n = len(test_list)
    if n == 0 :
        return 0.0
    transformedX = map(lambda x : fourth_moment_term(x,mu,sigma), test_list)
    sumX = reduce(lambda x, y : x + y, transformedX)
    return (sumX/float(n)) - 3
    
test_list = [1,2,3] 
mu = my_mean(test_list)
print(mu)
sigma = my_stdev(test_list, mu)
print(sigma)
gamma_1 = my_skew(test_list, mu, sigma)
print(gamma_1)
gamma_2 = my_kurtosis(test_list, mu, sigma)
print(gamma_2)




