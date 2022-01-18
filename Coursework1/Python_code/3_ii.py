# -*- coding: utf-8 -*-
# @File    : 3_ii.py
# @Author  : Zichi Zhang
# @Date    : 2021/12/17
# @Software: PyCharm

import numpy as np
import cvxpy as cp
import scipy as sp
import control as ctrl
import polytope as pc
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import scipy.linalg as spla
from IPython.core.pylabtools import figsize

mpl.rcParams['text.usetex'] = True  # 全局开启
mpl.rcParams['font.family'] = 'Times New Roman'  # 指定字体
figsize(12, 12)

# plt.rcParams['font.size'] = '14'
# Define the problem data
# -----------------------------
n = 2  # 2 states
m = 1  # 1 input
A = np.array([[1, 0.7], [-0.1, 1]])
B = np.array([[1], [0.5]])
N = 10

# Given matrices Q and R, solve DARE and determine
# a stabilising matrix K
# -----------------------------
Q = (2 ** 0.5) * np.eye(2)
R = 2 ** 0.5
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

# Define H_x, b_x, H_u and b_u
# -----------------------------
H_x = np.vstack((np.eye(2), -np.eye(2)))
b_x = np.vstack((x_max, -x_min))
H_u = np.array([[1], [-1]])
b_u = np.vstack((u_max, -u_min))

# Define X_kappa_f
# -----------------------------
H_kappa_f = np.vstack((H_x, H_u @ K))
b_kappa_f = np.vstack((b_x, b_u))
X_kappa_f = pc.Polytope(H_kappa_f, b_kappa_f)


# print(H_kappa_f)
# print(b_kappa_f)
# X_kappa_f.plot()
# plt.xlim([-2.2, 2.2])
# plt.ylim([-2.2, 2.2])
# plt.show()


def next_polytope(poly_j, poly_kappa_f, a_closed_loop):
    """
    Function:
        calculate the next polytope
    Inputs:
        poly_j        : the previous polytope
        poly_kappa_f  : the initial poly kappa f
        a_closed_loop : \dot{x}=Ax+B@Ku
    Returns:
        pc.Polytope(Hnext, bnext) : the next polytope
    """
    (Hj, bj) = (poly_j.A, poly_j.b)
    (Hkf, bkf) = (poly_kappa_f.A, poly_kappa_f.b)
    Hnext = np.vstack((Hkf, Hj @ a_closed_loop))
    bnext = np.concatenate((bkf, bj))
    return pc.Polytope(Hnext, bnext)


def determine_maximal_invariant_set(poly_kappa_f, a_closed_loop):
    """
    Function:
        determine the maximal invariant set
    Inputs:
        poly_kappa_f  : the initial poly kappa f
        a_closed_loop : \dot{x}=Ax+B@Ku
    Returns:
        inv_next      : the maximal invariant set
    """

    inv_prev = poly_kappa_f  # use the initial poly kappa f as the previous one before the loop
    keep_running = True
    while keep_running:  # loop to calculate the maximal set
        inv_next = next_polytope(inv_prev, poly_kappa_f, a_closed_loop)  # calculate the next one
        inv_next = pc.reduce(inv_next)
        keep_running = inv_next >= inv_prev  # if next one >= previous one, continue
        inv_prev = inv_next
    return inv_next


mis = determine_maximal_invariant_set(X_kappa_f, Acl)  # the maximal invariant set in this case
fig = plt.figure(2)
ax = fig.add_subplot(1, 1, 1)
(H_infty, b_infty) = (mis.A, mis.b)

H_z0 = spla.block_diag(H_x, H_u)
b_z0 = np.concatenate((b_x, b_u))
Hz1_last_row_block = np.hstack((H_infty @ A, H_infty @ B))  # the last row block of H_{z,1}
H_zi = np.vstack((spla.block_diag(H_x, H_u), Hz1_last_row_block))  # H_{z,1}
b_zi = np.concatenate((b_x, b_u, np.reshape(b_infty, (-1, 1))))  # b_{z,1}
S_i = pc.Polytope(H_zi, b_zi)
X_i = S_i.project([1, 2])

X_i.plot(ax=ax, color="cyan")
plt.xlim([-2.2, 2.2])
plt.ylim([-2.2, 2.2])
mis.plot(ax=ax)
plt.xlim([-2.2, 2.2])
plt.ylim([-2.2, 2.2])
X_i_list = [mis, X_i]
for i in range(N - 2):
    X_i_last = X_i
    Hzi_last_row_block = np.hstack((X_i.A @ A, X_i.A @ B))  # the last row block of H_{z,i}
    H_zi = np.vstack((H_z0, Hzi_last_row_block))  # H_{z,i}
    b_zi = np.concatenate((b_z0, np.reshape(X_i.b, (-1, 1))))  # b_{z,i}
    S_i = pc.Polytope(H_zi, b_zi)
    X_i = S_i.project([1, 2])
    X_i_list.append(X_i)
    # print("Is X_%d larger than X_%d?" % (i + 2, i + 1), X_i >= X_i_last)
color_list = ["pink", "pink", "pink", "pink", "cyan", "cyan", "white", "green", "blue", "pink"]
edgecolor_list = ["mediumseagreen", "orange", "black", "pink", "black", "black", "black", "black", "black", "pink"]
linestyle_list = ['-', '--', '--', '-', '-', '-', '-', '-', '-', '-']
linewidth_list = [5, 5, 5, 0, 0, 0, 0, 0, 0, 0]
plt.xticks(np.arange(-3, 3, 1), size=35)
plt.yticks(np.arange(-3, 3, 1), size=35)
for i in range(N):
    X_i_list[N - i - 1].plot(ax=ax, color=color_list[N - i - 1], linestyle=linestyle_list[N - i - 1],
                             linewidth=linewidth_list[N - i - 1], edgecolor=edgecolor_list[N - i - 1])
    plt.xlim([-2.2, 2.2])
    plt.ylim([-2.2, 2.2])
# 隐藏x、y轴的刻度线
ax = plt.gca()
# ax.xaxis.set_major_locator(x_major_locator)
plt.tick_params(bottom=False, top=False, left=False, right=False)
# plt.grid(axis='both', color='0.85', zorder=0)  # 画网格灰线
plt.xlabel(r"$x_1$", size=35, labelpad=5)
plt.ylabel(r"$x_2$", size=35, labelpad=15)
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width, box.height])
legend_elements = [Patch(facecolor='pink', edgecolor=edgecolor_list[3], linestyle=linestyle_list[3], linewidth=linewidth_list[3],
                         label=r"$X_{10}$"),
                   Patch(facecolor='white', edgecolor=edgecolor_list[2], linestyle=linestyle_list[2], linewidth=linewidth_list[2],
                         label=r"$X_2$"),
                   Patch(facecolor='white', edgecolor=edgecolor_list[1], linestyle=linestyle_list[1], linewidth=linewidth_list[1],
                         label=r"$X_1$"),
                   Patch(facecolor='white', edgecolor=edgecolor_list[0], linestyle=linestyle_list[0], linewidth=linewidth_list[0],
                         label=r"$O_{\infty}$"),
                   # Line2D([0], [0], marker='o', color='dodgerblue', label=r"$\mathrm{MPC}$",
                   #        markerfacecolor='none', markersize=15, mew=2, lw=3)
                   ]
# Create the legend
ax.legend(handles=legend_elements, bbox_to_anchor=(1, 1), prop={'size': 30}, edgecolor='black')
# grid
grid_xy = [-2, -1, 0, 1, 2]
for i in range(5):
    plt.plot(np.linspace(-2.2, 2.2, 1000), np.linspace(grid_xy[i], grid_xy[i], 1000), lw=2, color="#969696",
             alpha=0.5, zorder=-1)
    plt.plot(np.linspace(grid_xy[i], grid_xy[i], 1000), np.linspace(-2.2, 2.2, 1000), lw=2, color="#969696",
             alpha=0.5, zorder=-1)
# 隐藏上边和右边的框
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
# plt.tight_layout()
fig_adjust = 0.2
fig.subplots_adjust(left=fig_adjust, right=1-fig_adjust, top=1-fig_adjust, bottom=fig_adjust, wspace=1, hspace=1)
# Problem statement
# -----------------------------
x0 = cp.Parameter(n)  # <--- x is a parameter of the optimisation problem P_N(x)
u_seq = cp.Variable((m, N))  # <--- sequence of control actions
x_seq = cp.Variable((n, N + 1))

cost = 0
constraints = [x_seq[:, 0] == x0]  # x_0 = x
x_min = np.array([-2, -2])
x_max = np.array([2, 2])
u_min = np.array([-1])
u_max = np.array([1])
for t in range(N - 1):
    xt_var = x_seq[:, t]  # x_t
    ut_var = u_seq[:, t]  # u_t
    cost += cp.norm2(xt_var) ** 2 + ut_var ** 2

    # dynamics, x_min <= xt <= x_max, u_min <= ut <= u_max
    constraints += [x_seq[:, t + 1] == A @ xt_var + B @ ut_var,
                    x_min <= xt_var,
                    xt_var <= x_max,
                    u_min <= ut_var,
                    ut_var <= u_max]
# cost += 0     # the terminal cost V_f(x) = 0
problem = cp.Problem(cp.Minimize(cost), constraints)


def mpc(state):
    x0.value = state
    out = problem.solve()
    return u_seq[:, 0].value


# Solve the problem with MPC
# with 3 different extreme points
# -----------------------------
x_init = pc.extreme(X_i)[3]  # the extreme points of X_N
x_current = x_init
N_sim = 40
u_cache0 = []  # a list to save u_mpc
x_cache0 = x_current  # a list to save x_t
V_N_cache0 = []  # a list to save the cost value V_N^star
for t in range(N_sim):
    u_mpc = mpc(x_current)
    if t <= 20:
        u_cache0.append(u_mpc)
    x_current = A @ x_current + B @ u_mpc
    if t <= 19:
        x_cache0 = np.concatenate((x_cache0, x_current))
    if t <= 10:
        V_N_cache0.append(cost.value)
x_cache0 = np.reshape(x_cache0, (-1, n))

x_init = pc.extreme(X_i)[1]  # the extreme points of X_N
x_current = x_init
N_sim = 40
u_cache1 = []  # a list to save u_mpc
x_cache1 = x_current  # a list to save x_t
V_N_cache1 = []  # a list to save the cost value V_N^star
for t in range(N_sim):
    u_mpc = mpc(x_current)
    if t <= 20:
        u_cache1.append(u_mpc)
    x_current = A @ x_current + B @ u_mpc
    if t <= 19:
        x_cache1 = np.concatenate((x_cache1, x_current))
    if t <= 10:
        V_N_cache1.append(cost.value)
x_cache1 = np.reshape(x_cache1, (-1, n))

x_init = pc.extreme(X_i)[4]  # the extreme points of X_N
x_current = x_init
N_sim = 40
u_cache2 = []  # a list to save u_mpc
x_cache2 = x_current  # a list to save x_t
V_N_cache2 = []  # a list to save the cost value V_N^star
for t in range(N_sim):
    u_mpc = mpc(x_current)
    if t <= 20:
        u_cache2.append(u_mpc)
    x_current = A @ x_current + B @ u_mpc
    if t <= 19:
        x_cache2 = np.concatenate((x_cache2, x_current))
    if t <= 10:
        V_N_cache2.append(cost.value)
x_cache2 = np.reshape(x_cache2, (-1, n))

# plot states position
# plt.plot(x_cache0[:, 0], x_cache0[:, 1], marker='o', markersize=15, mew=2, lw=3, markerfacecolor='none', color='dodgerblue')
# plt.plot(x_cache1[:, 0], x_cache1[:, 1], marker='o', markersize=15, mew=2, lw=3, markerfacecolor='none', color='dodgerblue')
# plt.plot(x_cache2[:, 0], x_cache2[:, 1], marker='o', markersize=15, mew=2, lw=3, markerfacecolor='none', color='dodgerblue')

# plot x_t and u_t vs time
# ----------------------------
figsize(9, 6)
plt.rcParams['font.size'] = '14'
plt.figure(1)
plt.subplot(3, 1, (1, 2))
x_major_locator = plt.MultipleLocator(1)
ax = plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
plt.tick_params(bottom=False, top=False, left=False, right=False)
ax.tick_params(bottom=False, top=False, left=False, right=False)
ax.axes.xaxis.set_ticklabels([])
plt.xticks(np.arange(0, 21, 2), size=20)
plt.yticks(np.arange(-3, 3, 1), size=20)
plt.plot(x_cache0[:, 0], label=r"$x_{1}$", marker='o', markerfacecolor='none', markersize=7, mew=2, color='yellowgreen', linewidth=2, alpha=1, clip_on=False, zorder=100)  # plot x_1
plt.plot(x_cache0[:, 1], label=r"$x_{2}$", marker='x', markersize=7, mew=2, color='hotpink', linewidth=2, alpha=1, clip_on=False, zorder=100)  # plot x_2
plt.plot(2*np.ones(21), color='r', linestyle=(0, (2.5, 2.5)), linewidth=2, label=r"$x_{\max}$", zorder=50)  # plot x_max
plt.ylabel(r"$x_t$", size=20, labelpad=15)
plt.xlim(0, 20)
plt.grid(axis='both', color='0.85')
plt.legend(edgecolor='black', loc="upper right").set_zorder(150)
# plt.title('control actions vs time')
plt.subplot(3, 1, 3)
plt.xlabel(r"$t$", size=20)
plt.ylabel(r"$u_t$", size=20, labelpad=15)
plt.grid(axis='both', color='0.85')
x_major_locator = plt.MultipleLocator(1)
ax = plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
plt.tick_params(bottom=False, top=False, left=False, right=False)
ax.tick_params(bottom=False, top=False, left=False, right=False)
plt.xticks(np.arange(0, 21, 2), size=20)
plt.yticks(np.arange(-1, 1, 0.5), size=20)
plt.xlim([0, 20])
plt.ylim([-1.1, 0.5])
plt.tight_layout()
plt.plot(u_cache0, marker='o', markerfacecolor='none', markersize=7, mew=2, color='deepskyblue', linewidth=2, alpha=1, clip_on=False, zorder=100)
plt.plot(-np.ones(21), color='r', linestyle=(0, (2.5, 2.5)), linewidth=2, label=r"$u_{\min}$", zorder=50)  # plot u_min
plt.legend(edgecolor='black', loc="upper right").set_zorder(150)

# plot V_N^{\star} vs time
# ----------------------------
figsize(9, 4.5)
plt.figure(3)
# plt.title(r"$V_N^\star\;vs\;time$")
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
plt.xlim([0, 10])
plt.ylim(0, 25)
plt.tight_layout()
plt.plot(V_N_cache0, marker='X', markersize=6, color='#FF8C00', linewidth=1.5, alpha=1, clip_on=False, zorder=100)
plt.rc('axes', axisbelow=True)
plt.show()
