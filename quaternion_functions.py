import numpy as np

def vec2quat(vec):
    # exp mapping
    r = vec/2
    q = quat_exp([0,r[0],r[1],r[2]])
    return q

def quat2vec(q):
    # log mapping
    r = (2*quat_log(q))[1:]
    return r

def quat_norm(q):
    return np.linalg.norm(q)

def quat_normalize(q):
    q_norm = np.copy(q)
    q_norm /= quat_norm(q_norm)
    return q_norm

def quat_conjugate(q):
    q_conj = np.copy(q) # directly work on q won't change
    q_conj[1:] *= -1
    return q_conj

def quat_inverse(q):
    q_conj = quat_conjugate(q)
    q_norm = quat_norm(q)
    return q_conj/(q_norm**2)

def quat_multiply(p,q):
    p0 = p[0]
    pv = p[1:]
    q0 = q[0]
    qv = q[1:]

    r0 = p0*q0 - np.dot(pv, qv)
    rv = p0*qv + q0*pv + np.cross(pv, qv)

    return np.array([r0,rv[0],rv[1],rv[2]])

def quat_exp(q):
    q0 = q[0]
    qv = q[1:]
    qvnorm = np.linalg.norm(qv)

    z0 = np.exp(q0) * np.cos(qvnorm)
    if qvnorm == 0:
        zv = np.zeros(3)
    else:
        zv = np.exp(q0) * (qv / qvnorm) * np.sin(qvnorm)
    return np.array([z0, zv[0], zv[1], zv[2]])

def quat_log(q):
    qnorm = quat_norm(q)
    q0 = q[0]
    qv = q[1:]
    qvnorm = np.linalg.norm(qv)

    z0 = np.log(qnorm)
    if qvnorm == 0:
        zv = np.zeros(3)
    else:
        zv = (qv / qvnorm) * np.arccos(q0 / qnorm)
    return np.array([z0, zv[0], zv[1], zv[2]])


def quat_avg(q_set, qt):
    n = q_set.shape[0]

    epsilon = 1E-3
    max_iter = 1000
    for t in range(max_iter):
        err_vec = np.zeros((n, 3))
        for i in range(n):
            # calc error quaternion and transform to error vector
            qi_err = quat_normalize(quat_multiply(q_set[i, :], quat_inverse(qt)))
            vi_err = quat2vec(qi_err)

            # restrict vector angles to (-pi, pi]
            vi_norm = np.linalg.norm(vi_err)
            if vi_norm == 0:
                err_vec[i,:] = np.zeros(3)
            else:
                err_vec[i,:] = (-np.pi + np.mod(vi_norm + np.pi, 2 * np.pi)) / vi_norm * vi_err

        # measure derivation between estimate and real, then update estimate
        err = np.mean(err_vec, axis=0)
        qt = quat_normalize(quat_multiply(vec2quat(err), qt))

        if np.linalg.norm(err) < epsilon:
            break

    return qt, err_vec

# def quat_avg(q_set, qt):
#     n = q_set.shape[0]
#
#     q_square = np.zeros((4,4))
#     for i in range(n):
#         q_square += np.outer(q_set[i,:],q_set[i,:])
#     q_square /= n
#
#     # diagnoze
#     eigen_val, eigen_vec = np.linalg.eig(q_square)
#     qt = quat_normalize(eigen_vec[eigen_val.argmax()])
#
#     err_vec = np.zeros((n, 3))
#     mean_vec = quat2vec(qt)
#     for i in range(n):
#         ei_vec = quat2vec(q_set[i, :]) - mean_vec
#
#         # restrict vector angles to (-pi, pi]
#         evi_norm = np.linalg.norm(ei_vec)
#         if evi_norm == 0:
#             err_vec[i,:] = np.zeros(3)
#         else:
#             err_vec[i,:] = (-np.pi + np.mod(evi_norm + np.pi, 2 * np.pi)) / evi_norm * ei_vec
#
#     return qt, err_vec
