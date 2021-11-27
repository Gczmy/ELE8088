# -*- coding: utf-8 -*-
# @File    : Question3.py
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
A = np.array([[0.9, 1.5], [1.3, -0.7]])
B = np.array([[0.5], [0.2]])
Q = np.eye(2)      # Q = I2
R = 25
N = 40

# Computation of matrices P and K
# -----------------------------
Pf = np.eye(2)      # Pf = I2
P = np.zeros((n, n, N + 1))       # tensor
K = np.zeros((m, n, N))     # tensor (3D array)
P[:, :, N] = Pf

# Loop to calculate Pt by DP
# -----------------------------
for i in range(N):
    P_curr = P[:, :, N - i]
    K[:, :, N - i - 1] = -sp.linalg.solve(R + B.T @ P_curr @ B, B.T @ P_curr @ A)
    P[:, :, N - i - 1] = Q + A.T @ P_curr @ A + A.T @ P_curr @ B @ K[:, :, N - i - 1]
    # error = P_curr - P[:, :, N - i - 1]
    # print(np.linalg.norm(error, np.inf))
    # print(P[:, :, N - i - 1])

# determine an optimal stationary control law, i.e., determine K
P_N = P[:, :, 0]      # solves DARE
P_N, _, K = ctrl.dare(A, B, Q, R)
K = -K
print('K=')
print(K)

# Simulate the closed-loop system with the above controller:
# Simulate x(t+1) = Ax(t) + Bu(t), where u(t) = Kx(t), i.e.,
# simulate x(t+1) = (A+BK)x(t) --- start from any initial state
x = np.array([[19], [-2]])
u0 = K[0].T @ x     # different with Dr P.S: u0 = K @ x
u_star = u0
x_star = x
ut = []     # a list of ut
Vt = []     # a list of Vt
xt = x      # a matrix to save xt
for t in range(N - 1):
    u_star = K @ x_star     # u(t) = Kx(t)
    x_star = A @ x_star + B @ u_star        # x(t+1) = (A+BK)x(t)
    Vi = (0.5 * x_star.T @ P[:, :, N - i - 1] @ x_star)[0, 0]
    ut.append((u_star[0])[0])       # save the u(t)'s value every loop
    xt = np.hstack((xt, x_star))       # save the x(t+1)'s value every loop
    Vt.append(Vi)       # save the V*'s value every loop

# Plotting of solution
# -----------------------------
plt.rcParams['font.size'] = '14'

plt.figure(1)
plt.title('ut vs t')
plt.xlabel('time')
plt.ylabel('control action')
plt.plot(ut, label='ut')      # plot with time of x, control action of y
plt.legend()        # show labels

plt.figure(2)
plt.title('xt vs time')
plt.xlabel('time')
plt.ylabel('states')      # plot with time of x, states of y
plt.plot(xt[0, :], label='x1')
plt.plot(xt[1, :], label='x2')
plt.legend()        # show labels

plt.figure(3)
plt.title('Vt vs time')
plt.xlabel('time')
plt.ylabel('Vt')      # plot with time of x, Vt of y
plt.plot(Vt, label='Vt')
plt.legend()        # show labels
plt.show()
