# -*- coding: utf-8 -*-
# @File    : 3_ii.py
# @Author  : Zichi Zhang
# @Date    : 2022/3/12
# @Software: PyCharm

import numpy as np
import control as ctrl
import matplotlib.pyplot as plt

# Define the parameters
A_t = np.array([[1, 0],
                [-1, 2]])
B_t = np.array([[1],
                [0.5]])
G_t = np.array([[0.10, 0.05],
                [0.03, 0.20]])
C_t = np.array([[1, 0]])
D_t = np.array([[-1, 0.5]])

# Initial conditions
Sigma_w = np.array([[1.1, 0.2],
                    [0.2, 1.5]])  # the covariance matrix of w_t
Q_t = Sigma_w
R_t = 0.84  # the variance of v_t
E_x0 = np.array([[2],
                 [1]])  # the expectation of x_0
P_0 = np.array([[2, 0],
                [0, 4]])  # the variance of x_0

# The discrete-time control system
w = np.random.multivariate_normal(np.zeros(2), Q_t)
w = np.reshape(w, (2, 1))
v = np.random.normal(0, R_t)
x_0 = np.random.multivariate_normal((2, 1), P_0)
x_0 = np.reshape(x_0, (2, 1))

# check Riccati recursion
controllable_matrix = ctrl.ctrb(A_t.T, C_t.T)
controllable_matrix = np.linalg.matrix_rank(controllable_matrix)
print("(A, B) ctrl matrix rank:")
print(controllable_matrix)
C = np.linalg.cholesky(G_t @ Q_t @ G_t.T)
print("G_t @ Q_t @ G_t.T:")
print(G_t @ Q_t @ G_t.T)
print("C @ C.T:")
print(C @ C.T)
observability_matrix = ctrl.obsv(A_t, C)
obsv_matrix_rank = np.linalg.matrix_rank(observability_matrix)
print("(A, C) obsv matrix rank:")
print(obsv_matrix_rank)

def state_update(x, u_t):
    """
    Computes Ax + Bu + Gw
    """
    x_next = A_t @ x + B_t * u_t + G_t @ w
    return x_next


def output(x):
    """
    Computes y = Cx + Dw + v
    """
    y_t = C_t @ x + D_t @ w + v
    return y_t[0, 0]


def measurement_update(y_t, x_predicted, sigma_predicted):
    """
    Measurement update of the Kalman filter
    Returns the corrected state and covariance estimates after the output
    is measured
    """
    F = C_t @ sigma_predicted @ C_t.T + D_t @ Sigma_w @ D_t.T + R_t
    output_error = y_t - C_t @ x_predicted
    x_corrected = x_predicted + sigma_predicted @ C_t.T @ np.linalg.inv(F) @ output_error
    sigma_corrected = sigma_predicted - sigma_predicted @ C_t.T @ np.linalg.inv(F) @ C_t @ sigma_predicted
    return x_corrected, sigma_corrected


def time_update(x_meas_update, u_t, sigma_meas_update):
    """
    Measurement update of the Kalman filter
    Don't forget the input!
    """
    x_predicted = A_t @ x_meas_update + B_t * u_t
    sigma_predicted = A_t @ sigma_meas_update @ A_t.T + G_t @ Q_t @ G_t.T
    return x_predicted, sigma_predicted


# Simulate the system starting from a random initial state
t_sim = 80
x_actual_cache = np.zeros(shape=(t_sim+1, 2))
x_corr_cache = np.zeros(shape=(t_sim, 2))
x_pred_cache = np.zeros(shape=(t_sim+1, 2))

sigma_pred_cache = np.zeros(shape=(2, 2, t_sim+1))  # tensor
sigma_corr_cache = np.zeros(shape=(2, 2, t_sim))
x_pred = E_x0
sigma_pred = P_0

x_actual = x_0
x_actual_cache[0, :] = x_actual.T
x_pred_cache[0, :] = x_pred.T  # <- don't forget the .T
sigma_pred_cache[:, :, 0] = P_0


for t in range(t_sim):
    # --- OUTPUT MEASUREMENT
    # FLIP A "COIN" FIRST TO DECIDE WHETHER THERE IS AN OUTPUT
    # IF NOT, THE MEASUREMENT UPDATE CANNOT BE USED
    w = np.random.multivariate_normal(np.zeros(2), Q_t)
    w = np.reshape(w, (2, 1))
    v = np.random.normal(0, R_t)
    y = output(x_actual)
    u = 0

    # --- MEASUREMENT UPDATE
    x_corr, sigma_corr = measurement_update(y, x_pred, sigma_pred)
    x_corr_cache[t, :] = x_corr.T
    sigma_corr_cache[:, :, t] = sigma_corr

    # --- TIME UPDATE
    x_pred, sigma_pred = time_update(x_corr, u, sigma_corr)
    x_pred_cache[t, :] = x_pred.T
    sigma_pred_cache[:, :, t + 1] = sigma_pred

    # --- STATE UPDATE
    x_actual = state_update(x_actual, u)
    x_actual_cache[t + 1, :] = x_actual.T
    # print('(%d, %f)' % (t + 1, x_actual[1]))
    # print('(%d, %f)' % (t + 1, x_current[1]))

# Plotting of solution
# -----------------------------
t_sampling = 1/150
time_scale = np.arange(0, t_sim+1) * t_sampling
plt.figure(1)
plt.plot(time_scale[0:t_sim], x_corr_cache[:, 0], label="Corrected")
plt.plot(time_scale, x_actual_cache[:, 0], label="Actual pos")
# plt.plot(time_scale[0:t_sim], x_corr_cache[:, 1], label="Corrected")
# plt.plot(time_scale, x_actual_cache[:, 1], label="Actual pos")
# plt.plot(x_corr_cache[:, 0], x_corr_cache[:, 1], marker='*', label="Corrected")
# plt.plot(x_actual_cache[:, 0], x_actual_cache[:, 1], marker='x', label="Actual pos")
plt.xlabel("Time (s)")
plt.ylabel("Position (m)")
plt.legend()
plt.grid()
plt.figure(2)
plt.plot(time_scale, sigma_pred_cache[0, 0, :], label="sigma_pred(1, 1)")
plt.plot(time_scale, sigma_pred_cache[0, 1, :], label="sigma_pred(1, 2)")
plt.plot(time_scale, sigma_pred_cache[1, 0, :], label="sigma_pred(2, 1)")
plt.plot(time_scale, sigma_pred_cache[1, 1, :], label="sigma_pred(2, 2)")
plt.legend()
plt.grid()
plt.show()
