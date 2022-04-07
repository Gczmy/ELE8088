# -*- coding: utf-8 -*-
# @File    : 3_ii.py
# @Author  : Zichi Zhang
# @Date    : 2022/3/12
# @Software: PyCharm

import numpy as np
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
    f = C_t @ sigma_predicted @ C_t.T + D_t @ Sigma_w @ D_t.T + R_t
    output_error = y_t - C_t @ x_predicted
    x_corrected = x_predicted + sigma_predicted @ C_t.T @ output_error / f[0, 0]
    sigma_corrected = sigma_predicted - sigma_predicted @ C_t.T / f[0, 0] @ C_t @ sigma_predicted
    return x_corrected, sigma_corrected


def time_update(x_meas_update, u_t, sigma_meas_update):
    """
    Measurement update of the Kalman filter
    Don't forget the input!
    """
    x_predicted = A_t @ x_meas_update + B_t * u_t
    sigma_predicted = A_t @ sigma_meas_update @ A_t.T + G_t @ Q_t @ G_t.T
    return x_predicted, sigma_predicted

def update_F(sigma_predicted):
    """
    Measurement update of the Kalman filter
    Returns the corrected state and covariance estimates after the output
    is measured
    """
    f = C_t @ sigma_predicted @ C_t.T + D_t @ Sigma_w @ D_t.T + R_t
    F = sigma_predicted - sigma_predicted @ C_t.T / f[0, 0] @ C_t @ A_t
    return F

# Simulate the system starting from a random initial state
t_sim = 1002
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

F_cache = np.zeros(shape=(2, 2, t_sim))  # tensor
error_cache = np.zeros(shape=(t_sim, 2))

for t in range(t_sim):
    # --- OUTPUT MEASUREMENT
    # FLIP A "COIN" FIRST TO DECIDE WHETHER THERE IS AN OUTPUT
    # IF NOT, THE MEASUREMENT UPDATE CANNOT BE USED
    w = np.random.multivariate_normal(np.zeros(2), Q_t)
    w = np.reshape(w, (2, 1))
    v = np.random.normal(0, R_t)
    y = output(x_actual)
    u = 0.2

    # --- MEASUREMENT UPDATE
    x_corr, sigma_corr = measurement_update(y, x_pred, sigma_pred)
    # x_corr_cache[t, :] = x_corr.T
    sigma_corr_cache[:, :, t] = sigma_corr

    # --- TIME UPDATE
    x_pred, sigma_pred = time_update(x_corr, u, sigma_corr)
    # x_pred_cache[t, :] = x_pred.T
    sigma_pred_cache[:, :, t + 1] = sigma_pred

    # --- STATE UPDATE
    # x_actual = state_update(x_actual, u)
    # x_actual_cache[t + 1, :] = x_actual.T
    # print('(%d, %f)' % (t + 1, x_actual[1]))
    # print('(%d, %f)' % (t + 1, x_current[1]))

    # --- state estimation error
    F_cache[:, :, t] = update_F(sigma_pred)

print(f"F_0 = {F_cache[:, :, 0]}")
print(f"F_1 = {F_cache[:, :, 1]}")
print(f"F_1000 = {F_cache[:, :, 1000]}")
print(f"F_1001 = {F_cache[:, :, 1001]}")
