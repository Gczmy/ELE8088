# -*- coding: utf-8 -*-
# @File    : Question2.py
# @Author  : Zichi Zhang
# @Date    : 2021/11/17
# @Software: PyCharm

import numpy as np
import cvxpy as cp
import scipy as sp
import scipy.linalg as spla
import control as ctrl
import matplotlib.pyplot as plt

# Define the problem data
# -----------------------------
n = 2       # 2 states
m = 1       # 1 input
N = 50

A = np.array([[0.9, 1.5], [1.3, -0.7]])
B = np.array([[0.5], [0.2]])
Q = np.eye(2)      # Q = I2
R = 25

# Computation of matrices P and K
# -----------------------------
Pf = np.eye(2)      # Pf = I2
P = np.zeros((n, n, N + 1))       # tensor
K = np.zeros((m, n, N))     # tensor (3D array)
P[:, :, N] = Pf
P_N11 = []      # a list of P_{N,1,1}
P_N12 = []      # a list of P_{N,1,2}
P_N22 = []      # a list of P_{N,2,2}

# Loop to calculate Pt by DP
# -----------------------------
for i in range(N):
    P_curr = P[:, :, N - i]
    K[:, :, N - i - 1] = -sp.linalg.solve(R + B.T @ P_curr @ B, B.T @ P_curr @ A)
    P[:, :, N - i - 1] = Q + A.T @ P_curr @ A + A.T @ P_curr @ B @ K[:, :, N - i - 1]
    P_N11.append(P[0, 0, N - i - 1])        # save the P_{N,1,1}'s value every loop
    P_N12.append(P[0, 1, N - i - 1])        # save the P_{N,1,2}'s value every loop
    P_N22.append(P[1, 1, N - i - 1])        # save the P_{N,2,2}'s value every loop
    # error = P_curr - P[:, :, N - i - 1]
    # print(np.linalg.norm(error, np.inf))
    if i < 5:
        print('P_%d:' % (i+1))
        print(P[:, :, N - i - 1])
print('P_N:')
print(P[:, :, N - i - 1])

# determine an optimal stationary control law, i.e., determine K
P_N = P[:, :, 0]      # solves DARE
P_N, _, K = ctrl.dare(A, B, Q, R)
K = -K
print("P solves DARE:")
print(P_N)

# Plotting of solution
# -----------------------------
plt.title('P_N vs N')
plt.rcParams['font.size'] = '14'
plt.xlabel('N')
plt.ylabel('P_N')
plt.plot(P_N11, label='P_N11')      # plot with N of x, P_N11, P_N12, P_N22 of y
plt.plot(P_N12, label='P_N12')
plt.plot(P_N22, label='P_N22')
plt.legend()        # show labels
plt.show()
