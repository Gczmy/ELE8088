# -*- coding: utf-8 -*-
# @File    : Question1.py
# @Author  : Zichi Zhang
# @Date    : 2021/11/17
# @Software: PyCharm

import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

# Define the problem data
# -----------------------------
n = 1000    # Dimension of x
m = 100     # Dimension of b
x_zero = 0  # Number of zero elements of x
mu = 0

# Random data
# -----------------------------
x = cp.Variable(n)
np.random.seed(1)
A = np.random.rand(m, n)       # Let A be a matrix with random entries drawn from [0,1)
b = np.ones(m)      # b is a vector with b_i = 1 for i ∈ IN[1,m]

# Define some variables
# -----------------------------
sp_x = 0    # the sparsness of x
sp_x_cache = []     # a list of sp(x)'s value
mu_cache = []       # a list of mu's value
mu_max = cp.norm_inf(A.T @ b)       # mu's range is [0, mu_max]
Var_ii = cp.norm2(A @ x - b)      # var of question (ii)
Var_ii_cache = []       # a list of var of question (ii)

# Loop to solve the problem
# -----------------------------
for mu in np.linspace(0, mu_max.value, 10):     # divide [0, mu_max] to 10
    objective = cp.Minimize(0.5*cp.norm2(A @ x - b)**2 + mu * cp.norm1(x))      # optimization problem
    problem = cp.Problem(objective)
    problem.solve()
    for i in range(n):
        if x.value[i] < 10**(-7):      # if x < 10^(-7), we can consider that x = 0
            x_zero = x_zero + 1     # count the number of zero elements of x
    sp_x = x_zero/n             # the sparsness of x
    x_zero = 0                  # clear before next loop
    sp_x_cache.append(sp_x)     # save the sp(x)'s value every loop
    mu_cache.append(mu)         # save the mu's value every loop
    Var_ii_cache.append(Var_ii.value)
print("Error = %.2f" % problem.value)

# Plotting of solution
# -----------------------------
plt.figure(1)
plt.title('sp(x(mu)) vs mu')
plt.rcParams['font.size'] = '14'
plt.ylabel('sp(x(mu))')
plt.xlabel('mu')
plt.plot(mu_cache, sp_x_cache)      # plot with mu of x, sp(x(mu)) of y
plt.savefig('sp(x(mu))_vs_mu.png')

plt.figure(2)
plt.title('cp.norm2(A@x-b) vs mu')
plt.rcParams['font.size'] = '14'
plt.ylabel('cp.norm2(A@x-b)')
plt.xlabel('mu')
plt.plot(mu_cache, Var_ii_cache)      # plot with mu of x, sp(x(mu)) of y
plt.show()
plt.savefig('cp.norm2(A@x-b)_vs_mu.png')
