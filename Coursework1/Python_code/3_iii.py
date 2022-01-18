# -*- coding: utf-8 -*-
# @File    : 3_iii.py
# @Author  : Zichi Zhang
# @Date    : 2021/12/29
# @Software: PyCharm

import numpy as np
import cvxpy as cp
import scipy as sp
import control as ctrl
import polytope as pc
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.linalg as spla
from IPython.core.pylabtools import figsize

mpl.rcParams['text.usetex'] = True  # 全局开启
mpl.rcParams['font.family'] = 'Times New Roman'  # 指定字体
figsize(9, 6)

# Define the problem data
# -----------------------------
n = 2       # 2 states
m = 1       # 1 input
A = np.array([[1, 0.7], [-0.1, 1]])
B = np.array([[1], [0.5]])
N = 10

# Given matrices Q and R, solve DARE and determine
# a stabilising matrix K
# -----------------------------
Q = np.eye(2)
R = np.array([1])
P, _, K = ctrl.dare(A, B, Q, R)
K = -K
Acl = A + B @ K

# Define the sets of constraints; it is
# X = {x : H_x * x <= b_x}
# U = {u : H_u * u <= b_u}
# -----------------------------
x_min = np.array([[-2], [-2]])
x_max = np.array([[2], [2]])
u_min = np.array([-1])
u_max = np.array([1])

# Problem statement
# -----------------------------
x0 = cp.Parameter(n)        # <--- x is a parameter of the optimisation problem P_N(x)
u_seq = cp.Variable((m, N))     # <--- sequence of control actions
x_seq = cp.Variable((n, N+1))

cost = 0
constraints = [x_seq[:, 0] == x0]       # x_0 = x
x_min = np.array([-2, -2])
x_max = np.array([2, 2])
u_min = np.array([-1])
u_max = np.array([1])
for t in range(N-1):
    xt_var = x_seq[:, t]      # x_t
    ut_var = u_seq[:, t]      # u_t
    cost += cp.norm2(xt_var)**2 + ut_var**2

    # dynamics, x_min <= xt <= x_max, u_min <= ut <= u_max
    constraints += [x_seq[:, t+1] == A@xt_var + B@ut_var,
                    x_min <= xt_var,
                    xt_var <= x_max,
                    u_min <= ut_var,
                    ut_var <= u_max]
# cost += 0     # the terminal cost V_f(x) = 0

# Computation of matrices P and K
# -----------------------------
Pf = np.eye(2)      # Pf = I2
P = np.zeros((n, n, N + 1))       # tensor
K = np.zeros((m, n, N))     # tensor (3D array)
P[:, :, N] = Pf

# Calculate Pt by DP DARE
# -----------------------------
P = P[:, :, 0]      # solves DARE
P, _, K = ctrl.dare(A, B, Q, R)
K = -K
xN = x_seq[:, N-1]
cost += 0.5*cp.quad_form(xN, P)     # terminal cost

constraints += [x_min <= xN, xN <= x_max]       # terminal constraints (x_min <= xN <= x_max)
# Compute alpha using Equation
H = np.eye(2)
H = np.vstack((H, K))
H = np.vstack((H, -np.eye(2)))
H = np.vstack((H, -K))  # H = [[I], [K], [-I], [-K]]
b = np.array([[1], [1], [1], [1], [1], [1]])    # b = [[x_{max}], [u_{max}], [-x_{min}], [-u_{min}]]

# figure out P_N^(-1/2) Note:P_N is a matrix!
# v is the eigenvalue, Q is the eigenvector
v, Q = np.linalg.eig(P)
# print(v)
# V is the diagonal matrix of v
V = np.diag(v**(-0.5))
# print(V)
# P_N = Q * V * Q^(-1)
P_alpha = np.dot(np.dot(Q, V, np.linalg.inv(Q)), np.linalg.inv(Q))

# calculate the minimum of alpha
for i in range(6):
    alpha = 1/(np.linalg.norm(P_alpha @ np.reshape(H[i], (2, 1)))**2)
    if i == 0:
        alpha_temp = alpha
    if alpha < alpha_temp:
        alpha_temp = alpha
alpha = alpha_temp

# constraints of MPC
constraints_mpc = constraints + [cp.quad_form(xN, P) <= alpha]
problem = cp.Problem(cp.Minimize(cost), constraints_mpc)


def mpc(state):
    x0.value = state
    out = problem.solve()
    return u_seq[:, 0].value


# Solve the problem with MPC
# -----------------------------
# x_init = pc.extreme(X_i)[2]  # the extreme points of X_N
x_init = np.array([-2, 2])     # any feasible initial states
x_current = x_init

N_sim = 30
u_cache = []    # a list to save u_mpc
x_cache = x_current     # a list to save x_t
V_N_cache = []  # a list to save the cost value V_N^star
for t in range(N_sim):
    u_mpc = mpc(x_current)
    if t <= 20:
        u_cache.append(u_mpc)
    x_current = A @ x_current + B @ u_mpc
    if t <= 19:
        x_cache = np.concatenate((x_cache, x_current))
    if t <= 10:
        V_N_cache.append(cost.value)
x_cache = np.reshape(x_cache, (-1, n))

# Plotting of solution
# -----------------------------
plt.rcParams['font.size'] = '14'
# plot x_t and u_t vs time
# ----------------------------
plt.figure(1)
plt.subplot(3, 1, (1, 2))
# plt.title('States vs time')
x_major_locator = plt.MultipleLocator(1)
ax = plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
plt.tick_params(bottom=False, top=False, left=False, right=False)
ax.tick_params(bottom=False, top=False, left=False, right=False)
ax.axes.xaxis.set_ticklabels([])
plt.xticks(np.arange(0, 21, 2), size=20)
plt.yticks(np.arange(-3, 3, 1), size=20)
plt.plot(x_cache[:, 0], label=r"$x_{1}$", marker='o', markerfacecolor='none', markersize=7, mew=2, color='yellowgreen', linewidth=2, alpha=1, clip_on=False, zorder=100)  # plot x_1
plt.plot(x_cache[:, 1], label=r"$x_{2}$", marker='x', markersize=7, mew=2, color='hotpink', linewidth=2, alpha=1, clip_on=False, zorder=100)  # plot x_2
plt.plot(-2*np.ones(21), color='r', linestyle=(0, (2.5, 2.5)), linewidth=2, label=r"$x_{\min}$", zorder=50)  # plot x_min
plt.plot(2*np.ones(21), color='r', linestyle=(0, (2.5, 2.5)), linewidth=2, label=r"$x_{\max}$", zorder=50)  # plot x_max
plt.ylabel(r"$x_t$", size=20, labelpad=15)
plt.xlim(0, 20)
plt.grid(axis='both', color='0.85')
plt.legend(edgecolor='black', loc="upper right").set_zorder(150)        # show labels
# plt.title('control actions vs time')
plt.subplot(3, 1, 3)
ax = plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
plt.tick_params(bottom=False, top=False, left=False, right=False)
ax.tick_params(bottom=False, top=False, left=False, right=False)
plt.xticks(np.arange(0, 21, 2), size=20)
plt.yticks(np.arange(-1.5, 0.5, 0.5), size=20)
plt.plot(u_cache, marker='o', markerfacecolor='none', markersize=7, mew=2, color='deepskyblue', linewidth=2, alpha=1, clip_on=False, zorder=100)  # plot u
plt.plot(-np.ones(21), color='r', linestyle=(0, (2.5, 2.5)), linewidth=2, label=r"$u_{\min}$", zorder=50)  # plot u_min
plt.xlabel(r"$t$", size=20)
plt.ylabel(r"$u_t$", size=20, labelpad=15)
plt.grid(axis='both', color='0.85')
plt.legend(edgecolor='black', loc="upper right").set_zorder(150)        # show labels
plt.xlim(0, 20)

# plot V_N^{\star} vs time
# ----------------------------
figsize(9, 4.5)
plt.figure(2)
# plt.title(r"$V_N^{\star}$ vs time")
plt.xlim([0, 10])
plt.xlabel(r"$t$", size=20, labelpad=5)
plt.ylabel(r"$V_{10}^{\star}(x_t)$", size=20, labelpad=15)
plt.grid(axis='both', color='0.85')
x_major_locator = plt.MultipleLocator(1)
ax = plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
plt.tick_params(bottom=False, top=False, left=False, right=False)
ax.tick_params(bottom=False, top=False, left=False, right=False)
plt.xticks(np.arange(0, 11, 1), size=20)
plt.yticks(size=20)
plt.ylim(0, 25)
plt.tight_layout()
plt.plot(np.arange(11), V_N_cache, marker='X', markersize=6, color='#FF8C00', linewidth=1.5, alpha=1, clip_on=False, zorder=100)
plt.rc('axes', axisbelow=True)
plt.show()
