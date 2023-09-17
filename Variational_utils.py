## From 'A Convex Geodesic Selective Model for Image Segmentation', by Roberts, Michael, Ke Chen, and Klaus L. Irion 

from scipy.sparse.linalg import spsolve
import numpy as np
from scipy.ndimage import gaussian_filter, morphology
from numpy import linalg as LA
import numpy as np

from scipy import sparse
from scipy.sparse import spdiags
import torch


def get_relu_residual(aa_ts, bb_ts, eta, epsilon=0.02):

    aa_ts[aa_ts<epsilon] = 0.000001
    bb_ts[bb_ts<epsilon] = 0.000001
    bb_ts *= eta

    temp = aa_ts - bb_ts
    temp[temp<0] = 0
    temp = temp ** 2
    sq_err = (temp-torch.min(temp)) / (torch.max(temp) - torch.min(temp))
    sq_err[sq_err<epsilon] = 0.000001

    return sq_err


def Residual(new, old):
    
    Res = new.flatten() - old.flatten()
    R = LA.norm(Res) / LA.norm(old.flatten())
    return R

def AOS_v2(u, n, g, tau, mu, f, beta, dNu, b, alpha, varsigma, theta_tai):

    h = 1
    m = n
    N = m * n

    if np.sum(u.flatten()) == 0:
        print('Broken! All Zero')
        return


    T1 = np.logical_and(u <= varsigma, u >= -varsigma)
    T2 = np.logical_and(u <= 1 + varsigma, u >= 1 - varsigma)
    btaylor = np.logical_or(T1, T2).astype(np.int32) * b

    diags = np.array([0])

    bvecx = btaylor.flatten()
    bvecy = np.transpose(btaylor).flatten()

    A1 = 1.0 / (1.0 + tau * alpha * bvecx)
    Bx = spdiags(A1, diags, N, N)
    A2 = 1.0 / (1.0 + tau * alpha * bvecy)
    By = spdiags(A2, diags, N, N)

    R0 = np.transpose(-tau * f - tau * dNu)
    Rx = np.multiply(A1, R0.flatten())
    Ry = np.multiply(A2, np.transpose(R0).flatten())

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

    d1 = np.divide(1.0, np.sqrt((ux / h) ** 2 + (ucy / (2 * h)) ** 2 + beta * np.ones(np.shape(ucx))) )
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
    A1 = sparse.eye(N) * (1.0 - tau * theta_tai) - Bx * (2.0 * tau * mu * Ax)

    u_1 = (Rx + np.transpose(u).flatten()).astype(np.float32)
    u1 = spsolve(A1, u_1)
    u1 = np.transpose(u1.reshape((m, n)))

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
    u2 = u2.reshape((n, m))

    u = (u1 + u2) / 2

    return u





def ConvexVariational_Seg(z,lambda1,tau,Iters,utol,sigma1,lambda3,eps2,theta_tai,cols,rows,Mask):

    m, n = np.shape(z)
    SSF = np.zeros(np.shape(z))
    u = z.copy()

    if np.shape(cols)[0]*np.shape(cols)[1] == 1 and np.shape(rows)[0]*np.shape(rows)[1] == 1:
        Mask[rows:rows+2,cols:cols+2] = 1

    #### Additional parameters
    mu = 1 # regularisation term
    varsigma = 1e-2 # parameter in penalty function
    as1 = 2 # multiply minimum alpha value by as1
    beta = eps2 # parameter in curvature
    b = 161.7127690 # from Taylor expansion of fpen fn.

    sig = np.maximum(1.0, np.sqrt(100.0*sigma1))
    z_sm = gaussian_filter(z, sigma=sig)

    gy, gx = np.gradient(z_sm)
    nab_z = np.sqrt((gx**2)+(gy**2))
    beta1 = 10

    g = np.divide(1.0, (np.ones(np.shape(nab_z))+(beta1*(nab_z**2))))

    res = []

    c1 = np.sum(z[Mask>0.5])/np.sum(Mask>0.5)
    c2 = np.sum(z[Mask<0.5])/np.sum(Mask<0.5)
    f1 = (z-c1)**2-lambda3 *(z-c2)**2 + SSF

    for l in range(Iters):

        if (l+1) % 50 == 0:
            tau = np.maximum(1.0e-3,tau*0.9)

        oldu = u.copy()

        f0 = lambda1*f1/np.max(f1.flatten())
        f11 = f0.flatten()
        A = (1/2) * LA.norm(f11, np.inf)
        alpha = as1*A

        #### Calculate penalty term
        N1 = np.sqrt((2*u-1)**2+varsigma)-1
        Hnu = (1/2)+(1/np.pi)*np.arctan(N1/varsigma)
        dN_num1 = np.divide((4*u-2.0),(N1+1.0))
        dN = dN_num1 * (np.divide(varsigma*N1,np.pi*(varsigma**2+N1**2))+ Hnu)
        dNu = alpha*dN

        u = AOS_v2(u,m,g,tau,mu,f0,beta,dNu,b,alpha,varsigma,theta_tai)

        R = Residual(u,oldu)
        res.append(R)

        if R < utol:
            break

    return u
