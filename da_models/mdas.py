# -*- coding: utf-8 -*-
"""
Created on 2020/11/24 8:56

@author: John_Fengz
"""
import numpy as np
from sklearn.utils import check_random_state
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


def _centering(matrix):
    # matrix = matrix - np.mean(matrix)
    return matrix


class MDAS:
    def __init__(self, p, alpha=0.5, lamb1=0.1, lamb2=0.01, random_state=42):
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

        random_state : int, optional (default=42)
        """

        self.alpha = alpha
        self.lamb1 = lamb1
        self.lamb2 = lamb2
        self.p = p
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
        x_sou_ns = [_centering(x).T for x in x_sou_ns]
        x_sou_ds = [_centering(x).T for x in x_sou_ds]
        x_tar_ns = _centering(x_tar_ns).T

        K = len(x_sou_ds)  # num of source domains
        d, N_ds = x_sou_ds[0].shape  # num of features and ds instances
        if self.p is None:
            self.p = d
        eps = 0.0000000001

        def cal_H1(theta_):
            P_sub = [x_tar_ns - x for x in x_sou_ns]
            P_ = np.zeros((d, d))
            for k_ in range(K):
                P_ += theta_[k_] * P_sub[k_] @ P_sub[k_].T
            return P_

        def cal_H2(W_):
            Q_ = np.zeros((d, d))
            for k_ in range(K):
                for l_ in range(K):
                    Q_sub = x_sou_ds[k_] @ W_[k_] - x_sou_ds[l_]
                    Q_ += Q_sub @ Q_sub.T
            return Q_

        def cal_H(theta_, W_):
            return self.alpha * cal_H1(theta_) + (1 - self.alpha) * cal_H2(W_)

        # initialize all the parameters
        random_state = check_random_state(self.random_state)
        theta = np.ones(K).T / K
        W = [np.eye(N_ds, N_ds)] * K

        for _ in range(5):
            # update \mathcal{A}
            H = cal_H(theta, W)
            # calculate the eigenvectors and eigenvalues of H
            w, V = linalg.eig(H)
            ind = np.argsort(w)
            A = V[:, ind[:self.p]]

            # update \mathbf{\theta}
            u = [_frob_2(A.T @ (x_tar_ns - x)) for x in x_sou_ns]

            def obj_theta(x):
                return x @ u + self.lamb1 * np.linalg.norm(x, ord=2) ** 2

            ones = np.ones(K)
            bnds = tuple([(0, 1)] * K)
            cons = ({'type': 'eq', 'fun': lambda x: x @ ones - 1})
            res = minimize(obj_theta, theta.T, method='SLSQP', bounds=bnds, constraints=cons)
            theta = res.x.T

            # update \mathcal{W}
            C_sub = np.zeros_like(x_sou_ds[0])
            for l in range(K):
                C_sub += x_sou_ds[l]
            for k in range(K):
                B = K * x_sou_ds[k].T @ A @ A.T @ x_sou_ds[k]
                U = np.diag(0.5 / (np.sqrt(np.sum(W[k] * W[k], 1) + eps)))
                C = x_sou_ds[k].T @ A @ A.T @ C_sub
                W[k] = np.linalg.inv(B + self.lamb2 * U) @ C

            H = cal_H(theta, W)
            final_obj = np.trace(A.T @ H @ A) + self.lamb1 * np.linalg.norm(theta, ord=2) ** 2 + \
                self.lamb1 * _l21_norm(np.vstack(W))
            print(final_obj)
            self.objs.append(final_obj)

        self.A = A
        self.W = W
        x_tar_ds_new = [(A.T @ x @ y).T for (x, y) in zip(x_sou_ds, W)]
        return x_tar_ds_new

    def transform(self, x_new):
        x_new = x_new.T
        x_new_t = self.A.T @ x_new
        return x_new_t.T
