import numpy as np
from quaternion_functions import *

def compute_sigma_pts(q, P, Q):
    n = P.shape[0]

    # compute distribution around zero, apply noise before process model
    S = np.linalg.cholesky(P + Q)
    Xpos = S * np.sqrt(2*n)
    Xneg = -S * np.sqrt(2*n)
    W = np.hstack((Xpos, Xneg))

    # shift mean and transform to quaternions
    X = np.zeros((2*n, 4))
    for i in range(2*n):
        qW = vec2quat(W[:, i])
        X[i, :] = quat_multiply(q, qW)

    # add mean, 2n+1 sigma points in total
    X = np.vstack((q, X))

    return X

def process_model(X, gyro, dt):
    n = X.shape[0]
    Y = np.zeros((n,4))

    # compute delta quaternion
    qdelta = vec2quat(gyro*dt)

    for i in range(n):
        # project sigma points by process model
        q = X[i, :]
        Y[i, :] = quat_multiply(q, qdelta)

    return Y

def prediction(Y, qk):
    n = Y.shape[0]
    # compute mean (in quaternion)
    q_pred, W = quat_avg(Y, qk)

    # compute covariance (in vector)
    P_pred = np.zeros((3, 3))
    for i in range(n):
        P_pred += np.outer(W[i,:], W[i,:])
    P_pred /= n

    return q_pred, P_pred, W

def measurement_model(Y, acc, W, R):
    n = Y.shape[0]

    # define world gravity in quaternion
    g_q = np.array([0, 0, 0, 1])

    Z = np.zeros((n, 3))
    for i in range(n):
        # compute predicted acceleration in body-frame
        q = Y[i, :]
        Z[i, :] = quat_multiply(quat_multiply(quat_inverse(q), g_q), q)[1:] # rotate from world frame to body frame

    # measurement mean
    zk = np.mean(Z, axis=0)
    zk /= np.linalg.norm(zk)

    # measurement cov and correlation
    Pzz = np.zeros((3, 3))
    Pxz = np.zeros((3, 3))
    Z_err = Z - zk
    for i in range(n):
        Pzz += np.outer(Z_err[i,:], Z_err[i,:])
        Pxz += np.outer(W[i,:], Z_err[i,:])
    Pzz /= n
    Pxz /= n

    # innovation
    acc /= np.linalg.norm(acc)
    vk = acc - zk
    Pvv = Pzz + R

    return vk, Pvv, Pxz

def update(q_pred, P_pred, vk, Pvv, K):
    # note: q_pred, P_pred are in quaternion, while vk, Pvv in vector
    q_gain = vec2quat(K.dot(vk))
    q_update = quat_multiply(q_gain,q_pred)
    P_update = P_pred - K.dot(Pvv).dot(K.T)
    return q_update, P_update

