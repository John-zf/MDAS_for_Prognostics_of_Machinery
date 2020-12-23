# -*- coding: utf-8 -*-
"""
Created on 2020/11/24 8:56

@author: John_Fengz
"""
import numpy as np
from sklearn.utils import check_random_state
from sklearn.metrics.pairwise import linear_kernel, polynomial_kernel, rbf_kernel
from scipy.optimize import minimize
from scipy import linalg


def _frob_2(matrix):
    """
    Calculate the square frobenius norm of a matrix.
    Parameters
    ----------
    matrix : array-like (numpy.array).
    Returns
    -------
    frob_norm_2 : float
        the value of the square of frobenius norm.
    """

    frob_norm_2 = np.linalg.norm(matrix, 'fro') ** 2
    return frob_norm_2


def _l21_norm(matrix):
    """
    Calculate the l2,1 norm of a matrix.
    See : 'Nie et. al. Efficient and Robust Feature Selection via Joint l2,1-Norms Minimization'
    Parameters
    ----------
    matrix : array-like (numpy.array).
    Returns
    -------
    l21_norm : float
        the value of the l2,1 norm
    """

    l21_norm = np.sum(np.sqrt(np.sum(matrix * matrix, 1)))
    return l21_norm


def _kernel(x1, x2, kernel='linear'):
    """
    Calculate the kernel matrix of two datasets.
    Parameters
    ----------
    x1 : array-like (numpy.array).
    x2 : array-like (numpy.array).
    kernel : string (linear, rbf, poly)
    Returns
    -------
        Kernel matrix of two datasets
    """

    if kernel == 'linear':
        return linear_kernel(x1.T, x2.T)

    elif kernel == 'rbf':
        return rbf_kernel(x1.T, x2.T)

    elif kernel == 'poly':
        return polynomial_kernel(x1.T, x2.T)

    else:
        raise Exception('kernel must be one of linear, rbf, poly.')


def _normalize(matrix):
    matrix /= np.linalg.norm(matrix, axis=0)
    return matrix


class MDAS:
    def __init__(self, p, alpha=0.5, lamb1=0.1, lamb2=0.1, kernel='linear', dic=None, random_state=42):
        """
        Parameters
        ----------
        p : int
            Number of subspace.
        alpha : float, optional (default=0.5)
            Ratio to control the intra and inter consistency.
        lamb1 : float, optional (default=0.01)
            Regularization paramter.
        lamb2 : float, optional (default=0.01)
            Regularization paramter.
        p : int
            Number of subspace.
        dic : num_dic * d
            Dictionary of all the instances.
        random_state : int, optional (default=42)
        """
        self.p = p
        self.alpha = alpha
        self.lamb1 = lamb1
        self.lamb2 = lamb2
        self.kernel = kernel
        self.dic = dic
        self.random_state = random_state

        self.W = None
        self.A = None
        self.objs = []

    def fit(self, x_tar_ns, x_sou_ns, x_sou_ds):
        """
        Fit the MDAS model and return the transformed features.
        Parameters
        ----------
        x_tar_ns : ns * d
            Features of the target domain in the normal state.
        x_sou_ns : num_sd * ns * d
            A list of features of the source domain in the normal state.
        x_sou_ds : num_sd * ds * d
            A list of features of the source domain in the degenerated state.
        Returns
        -------
        k_tar_ds_new : Transformed features of the target domain in the degenerated state.
        """

        # preprocessing all the features
        x_sou_ns = [x.T for x in x_sou_ns]
        x_sou_ds = [x.T for x in x_sou_ds]
        x_tar_ns = x_tar_ns.T
        # precompute all the kernel matrices
        self.dic = self.dic.T
        self.dic = _normalize(self.dic)
        K_K = _kernel(self.dic, self.dic, self.kernel)
        K_tar_ns = _kernel(self.dic, x_tar_ns, self.kernel)
        K_sou_ns = [_kernel(self.dic, x, self.kernel) for x in x_sou_ns]
        K_sou_ds = [_kernel(self.dic, x, self.kernel) for x in x_sou_ds]
        K = len(K_sou_ds)  # num of source domains
        d, N_ds = K_sou_ds[0].shape  # num of features and ds instances
        if self.p is None:
            self.p = d
        eps = np.spacing(0)

        def cal_H1(theta_):
            P_sub = [K_tar_ns - x for x in K_sou_ns]
            P_ = np.zeros((d, d))
            for k_ in range(K):
                P_ += theta_[k_] * P_sub[k_] @ P_sub[k_].T
            return P_

        def cal_H2(W_):
            Q_ = np.zeros((d, d))
            for k_ in range(K):
                temp = K_sou_ds[k_] @ W_[k_]
                for l_ in range(K):
                    Q_sub = temp - K_sou_ds[l_]
                    Q_ += Q_sub @ Q_sub.T
            return Q_

        def cal_H(theta_, W_):
            return self.alpha * cal_H1(theta_) + (1 - self.alpha) * cal_H2(W_)

        def cal_final_obj(_A, _theta, _W):
            _H = cal_H(_theta, _W)
            obj = np.trace(_A.T @ _H @ _A) + self.lamb1 * np.linalg.norm(_theta, ord=2) ** 2 + \
                  self.lamb1 * _l21_norm(np.vstack(_W))
            return obj

        # initialize all the parameters
        random_state = check_random_state(self.random_state)
        theta = np.ones(K).T / K
        W = [random_state.randn(N_ds, N_ds)] * K
        A = None
        for _ in range(1):
            # update \mathcal{A}
            H = cal_H(theta, W)
            # calculate the eigenvectors and eigenvalues of H
            K_K_i = np.linalg.inv(K_K)
            e, V = linalg.eig(K_K_i @ H)
            e, V = e.real, V.real
            ind = np.argsort(e)
            A = V[:, ind[:self.p]]
            # update \mathbf{\theta}
            u = [_frob_2(A.T @ (K_tar_ns - x)) for x in K_sou_ns]

            def obj_theta(x):
                return x @ u + self.lamb1 * np.linalg.norm(x, ord=2) ** 2

            ones = np.ones(K)
            bnds = tuple([(0, 1)] * K)
            cons = ({'type': 'eq', 'fun': lambda x: x @ ones - 1})
            res = minimize(obj_theta, theta.T, method='SLSQP', bounds=bnds, constraints=cons)
            theta = res.x.T

            # update \mathcal{W}
            C_sub = np.zeros_like(K_sou_ds[0])
            for l in range(K):
                C_sub += K_sou_ds[l]
            for k in range(K):
                B = K * K_sou_ds[k].T @ A @ A.T @ K_sou_ds[k]
                U = np.diag(0.5 / (np.sqrt(np.sum(W[k] * W[k], 1) + eps)))
                C = K_sou_ds[k].T @ A @ A.T @ C_sub
                W[k] = np.linalg.inv(B + self.lamb2 * U) @ C

            final_obj = cal_final_obj(A, theta, W)
            print(final_obj)
            self.objs.append(final_obj)

        self.A = A
        self.W = W
        K_tar_ds_new = [(A.T @ x @ y).T for (x, y) in zip(K_sou_ds, W)]
        return K_tar_ds_new

    def transform(self, x_new):
        x_new = x_new.T
        K_x = _kernel(self.dic, x_new)
        x_new_t = self.A.T @ K_x
        return x_new_t.T
