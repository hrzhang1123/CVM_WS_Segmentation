

from scipy.sparse.linalg import spsolve

from numpy import linalg as LA
import numpy as np

from scipy import sparse
from scipy.sparse import spdiags
import torch


def get_relu_residual(aa_ts, bb_ts, eta, epsilon=0.02):

    aa_ts[aa_ts<epsilon] = 0.00001
    bb_ts[bb_ts<epsilon] = 0.00001

    bb_ts *= eta

    temp = aa_ts - bb_ts
    temp[temp<0] = 0
    temp = temp ** 2
    sq_err = (temp-torch.min(temp)) / (torch.max(temp) - torch.min(temp))
    sq_err[sq_err<epsilon] = 0.0000001

    return sq_err


def Residual(new, old):
    n, m = np.shape(new)
    Res = new.flatten() - old.flatten()
    R = LA.norm(Res) / LA.norm(old.flatten())
    return R

def AOS_v2(u, n, g, tau, mu, f, beta, dNu, b, alpha, varsigma, theta_tai):
    ####### CMS Spencer, Chen AOS2 %%%%%%%%%%

    h = 1
    m = n
    N = m * n

    if np.sum(u.flatten()) == 0:
        print('Broken! All Zero')
        return

    btaylor = np.zeros((n, n))

    T1 = np.logical_and(u <= varsigma, u >= -varsigma)
    T2 = np.logical_and(u <= 1 + varsigma, u >= 1 - varsigma)
    btaylor = np.logical_or(T1, T2).astype(np.int32) * b

    diags = np.array([0])
    ON = sparse.eye(N)

    bvecx = btaylor.flatten()
    taybx = spdiags(bvecx, diags, N, N)
    bvecy = np.transpose(btaylor).flatten()
    tayby = spdiags(bvecy, diags, N, N)

    A1 = 1.0 / (1.0 + tau * alpha * bvecx)
    Bx = spdiags(A1, diags, N, N)  # ON + tau*alpha*tayby
    A2 = 1.0 / (1.0 + tau * alpha * bvecy)
    By = spdiags(A2, diags, N, N)  # ON + tau*alpha*tayby

    R0 = np.transpose(-tau * f - tau * dNu)
    Rx = np.multiply(A1, R0.flatten())  # np.multiply(csr_matrix(Bx),R0)
    Ry = np.multiply(A2, np.transpose(R0).flatten())  # np.multiply(csr_matrix(Bx),R0)

    ux, uy = np.gradient(u)

    ucx = u[2:, :] - u[:n - 2, :]
    U11 = np.expand_dims(u[1, :] - u[0, :], 0)
    U12 = np.expand_dims(u[-1, :] - u[-2, :], 0)
    ucx = np.concatenate((U11, ucx), 0)
    ucx = np.concatenate((ucx, U12), 0)

    ucy = u[:, 2:] - u[:, :m - 2]
    U11 = np.expand_dims(u[:, 1] - u[:, 0], 1)
    U12 = np.expand_dims(u[:, -1] - u[:, -2], 1)
    ucy = np.concatenate((U11, ucy), 1)
    ucy = np.concatenate((ucy, U12), 1)

    d1 = np.divide(1.0, np.sqrt(((ux / h) ** 2) + ((ucy / (2 * h)) ** 2) + (beta * np.ones(np.shape(ucx)))))
    d2 = np.divide(1.0, np.sqrt(((ucx / (2 * h)) ** 2) + ((uy / h) ** 2) + (beta * np.ones(np.shape(ucx)))))

    d11 = g * d1
    a1 = np.zeros((N, 1))
    a2 = np.zeros((N, 1))
    a3 = np.zeros((N, 1))

    for j in range(m):
        j0 = j * n
        d_1 = np.expand_dims(d11[:n - 1, j], 1)
        a1[j0:n - 1 + j0] = +d_1
        a3[1 + j0:n + j0] = +d_1
        a2[j0:n - 1 + j0] = -d_1
        a2[1 + j0:n + j0] = a2[1 + j0:n + j0] - d_1

    AA11 = np.concatenate((np.transpose(a1), np.transpose(a2)), 0)
    AA11 = np.concatenate((AA11, np.transpose(a3)), 0)

    Ax = spdiags(AA11, np.array([-1, 0, 1]), N, N)
    A1 = sparse.eye(N) * (1.0 - tau * theta_tai) - Bx * (2.0 * tau * mu * Ax);

    u_1 = (Rx + np.transpose(u).flatten()).astype(np.float32)
    u1 = spsolve(A1, u_1)
    u1 = np.transpose(u1.reshape((n, m)))

    d22 = g * d2
    b1 = np.zeros((N, 1))
    b2 = np.zeros((N, 1))
    b3 = np.zeros((N, 1))
    for i in range(n):
        i0 = i * m
        d_2 = np.expand_dims(d22[i, :m - 1], 1)
        b1[i0:m - 1 + i0] = +d_2
        b3[1 + i0:m + i0] = +d_2
        b2[i0:m - 1 + i0] = -d_2
        b2[1 + i0:m + i0] = b2[1 + i0:n + i0] - d_2

    AA11 = np.concatenate((np.transpose(b1), np.transpose(b2)), 0)
    AA11 = np.concatenate((AA11, np.transpose(b3)), 0)

    Ay = spdiags(AA11, np.array([-1, 0, 1]), N, N)
    A2 = sparse.eye(N) * (1.0 - tau * theta_tai) - By * (2.0 * tau * mu * Ay)

    u_2 = (Ry + u.flatten()).astype(np.float32)
    u2 = spsolve(A2, u_2)
    u2 = u2.reshape((m, n))
    # u2 = np.transpose(u2)

    # u = u2
    u = (u1 + u2) / 2

    return u
