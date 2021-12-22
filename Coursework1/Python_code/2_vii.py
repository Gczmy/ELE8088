# -*- coding: utf-8 -*-
# @File    : 2_vii.py
# @Author  : Zichi Zhang
# @Date    : 2021/12/21
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

A = np.array([[1, 0], [1, 1]])
B = np.array([[1], [0]])
Q = np.eye(2)      # Q = I2
R = 2
S = np.array([[-0.5], [0.5]])

Q_tilde = Q - S * R**(-1) @ S.T
A_tilde = A - B * R**(-1) @ S.T

# Computation of matrices P and K
# -----------------------------
Pf_0 = np.array([[0, 0], [0, 0]])
Pf_1 = np.array([[1, 1], [1, 1]])
P_Pf_0 = np.zeros((n, n, N + 1))       # tensor
P_Pf_1 = np.zeros((n, n, N + 1))       # tensor
P_N = np.zeros((n, n, N + 1))       # tensor
K = np.zeros((m, n, N))     # tensor (3D array)

# Loop to calculate Pt by DP
# -----------------------------
# P0=[0,0;0,0]
P_Pf_0[:, :, N] = Pf_0
for i in range(N):
    P_curr = P_Pf_0[:, :, N - i]
    K[:, :, N - i - 1] = -sp.linalg.solve(R + B.T @ P_curr @ B, B.T @ P_curr @ A_tilde)
    P_Pf_0[:, :, N - i - 1] = Q_tilde + A_tilde.T @ P_curr @ A_tilde + A_tilde.T @ P_curr @ B @ K[:, :, N - i - 1]
    # error = P_curr - P[:, :, N - i - 1]
    # print(np.linalg.norm(error, np.inf))
print('P_N(P_0=[0,0;0,0]):')
print(P_Pf_0[:, :, N - i - 1])

# P0=[1,1;1,1]
P_Pf_1[:, :, N] = Pf_1
for i in range(N):
    P_curr = P_Pf_1[:, :, N - i]
    K[:, :, N - i - 1] = -sp.linalg.solve(R + B.T @ P_curr @ B, B.T @ P_curr @ A_tilde)
    P_Pf_1[:, :, N - i - 1] = Q_tilde + A_tilde.T @ P_curr @ A_tilde + A_tilde.T @ P_curr @ B @ K[:, :, N - i - 1]
    # error = P_curr - P[:, :, N - i - 1]
    # print(np.linalg.norm(error, np.inf))
print('P_N(P_0=[0,0;0,0]):')
print(P_Pf_1[:, :, 0])

# Loop to calculate Pt by Equation(35)---> P = A⊺PA − (A⊺PB + S)(R + B⊺PB)−1(B⊺PA + S⊺) + Q
# -----------------------------
P_N[:, :, N] = Pf_0
for i in range(N):
    P_curr = P_N[:, :, N - i]
    K[:, :, N - i - 1] = -sp.linalg.solve(R + B.T @ P_curr @ B, B.T @ P_curr @ A + S.T)
    P_N[:, :, N - i - 1] = Q + A.T @ P_curr @ A + (A.T @ P_curr @ B + S) @ K[:, :, N - i - 1]
    # error = P_curr - P[:, :, N - i - 1]
    # print(np.linalg.norm(error, np.inf))
print('P_N of Equation(35):')
print(P_N[:, :, 0])
