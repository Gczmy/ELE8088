# -*- coding: utf-8 -*-
# @File    : Question4.py
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
A = np.array([[0.9, 1.5], [1.3, -0.7]])   # todo:问为什么[[0.9, 1.5], [1.3, -0.7]]不行？
# A = np.array([[1, 1], [1, 0]])
B = np.array([[0.5], [0.2]])
Q = np.eye(2)      # Q = I2
R = 10
N = 40

x_min = np.array([-1, -1])
x_max = np.array([1, 1])
u_min = np.array([-1])
u_max = np.array([1])

# Problem statement
# -----------------------------
x0 = cp.Parameter(n)        # <--- x is a parameter of the optimisation problem P_N(x)
u_seq = cp.Variable((m, N))     # <--- sequence of control actions
x_seq = cp.Variable((n, N+1))

cost = 0
constraints = [x_seq[:, 0] == x0]       # x_0 = x

for t in range(N-1):
    xt_var = x_seq[:, t]      # x_t
    ut_var = u_seq[:, t]      # u_t
    cost += 0.5*(cp.quad_form(xt_var, Q) + R*ut_var**2)

    # dynamics, x_min <= xt <= x_max, u_min <= ut <= u_max
    constraints += [x_seq[:, t+1] == A@xt_var + B@ut_var,
                    x_min <= xt_var,
                    xt_var <= x_max,
                    u_min <= ut_var,
                    ut_var <= u_max]

# Computation of matrices P and K
# -----------------------------
Pf = np.eye(2)      # Pf = I2
P = np.zeros((n, n, N + 1))       # tensor
K = np.zeros((m, n, N))     # tensor (3D array)
P[:, :, N] = Pf

# Loop to calculate Pt by DP DARE
# -----------------------------
for i in range(N):
    P_curr = P[:, :, N - i]
    K[:, :, N - i - 1] = -sp.linalg.solve(R + B.T @ P_curr @ B, B.T @ P_curr @ A)
    P[:, :, N - i - 1] = Q + A.T @ P_curr @ A + A.T @ P_curr @ B @ K[:, :, N - i - 1]
    # error = P_curr - P[:, :, N - i - 1]
    # print(np.linalg.norm(error, np.inf))
    # print(P[:, :, N - i - 1])
P = P[:, :, 0]      # solves DARE
P, _, K = ctrl.dare(A, B, Q, R)
K = -K
# print(P)
xN = x_seq[:, N-1]
cost += 0.5*cp.quad_form(xN, P)     # terminal cost
constraints += [x_min <= xN, xN <= x_max]       # terminal constraints (x_min <= xN <= x_max)

# Compute alpha using Equation
H = np.eye(2)
H = np.vstack((H, K))
H = np.vstack((H, -np.eye(2)))
H = np.vstack((H, -K))      # H = [[I], [K], [-I], [-K]]
b = np.array([[1], [1], [1], [1], [1], [1]])        # b = [[x_{max}], [u_{max}], [-x_{min}], [-u_{min}]]

# 求P_N的(-1/2)次方
# v 为特征值    Q 为特征向量
v, Q = np.linalg.eig(P)
# print(v)
V = np.diag(v**(-0.5))
# print(V)
# P_N = Q * V * Q**-1
P = np.dot(np.dot(Q, V, np.linalg.inv(Q)), np.linalg.inv(Q))

# 求alpha的最小值
for i in range(6):
    alpha = 1/(np.linalg.norm(P @ np.reshape(H[i], (2, 1)))**2)     # todo:问为什么cp.norm2不行？
    if i == 0:
        alpha_temp = alpha
    if alpha < alpha_temp:
        alpha_temp = alpha
    alpha_temp = alpha
# print(alpha)

# # constraints of MPC
# constraints_mpc = constraints + [cp.quad_form(xN, P) <= alpha]
# problem = cp.Problem(cp.Minimize(cost), constraints_mpc)
AI = np.eye(2)

# constraints of MPC Q4 iii
constraints_mpc = constraints + [cp.norm_inf(AI@xt_var+B@ut_var) <= 0.05]
problem = cp.Problem(cp.Minimize(cost), constraints_mpc)


def mpc(state):
    x0.value = state
    out = problem.solve()
    return u_seq[:, 0].value


x_init = np.array([0.01, 0.04])
x_current = x_init

Nsim = 40
u_cache = []
x_cache = x_current
V_N_cache = []
for t in range(Nsim):
    u_mpc = mpc(x_current)
    u_cache.append(u_mpc)
    x_current = A @ x_current + B @ u_mpc
    x_cache = np.concatenate((x_cache, x_current))
    V_N_cache.append(cost.value)
x_cache = np.reshape(x_cache, (Nsim+1, n))

# Plotting of solution
# -----------------------------

plt.rcParams['font.size'] = '14'
plt.figure(1)
plt.title('States vs time')
plt.plot(x_cache[:, 0], label='x1')
plt.plot(x_cache[:, 1], label='x2')
plt.xlabel('Time, t')
plt.ylabel('States, x_t')
plt.legend()        # show labels

plt.figure(2)
plt.title('control actions vs time')
plt.plot(u_cache)
plt.xlabel('Time, t')
plt.ylabel('control actions, u_t')

plt.figure(3)
plt.title('States situation')
plt.xlim([-0.1, 0.1])
plt.ylim([-0.1, 0.1])
plt.plot(x_cache[:, 0], x_cache[:, 1], '-o')
plt.xlabel('x1')
plt.ylabel('x2')

plt.figure(4)
plt.title('V_N_star vs time')
plt.plot(V_N_cache)
plt.xlabel('Time, t')
plt.ylabel('V_N_star')
plt.show()
