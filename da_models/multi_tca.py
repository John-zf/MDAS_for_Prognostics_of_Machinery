# -*- coding: utf-8 -*-
"""
Created on 2020/12/29 11:04
Multi-Domain Transfer Component Analysis for Domain
Generalization
@author: John_Fengz
"""
import numpy as np
import scipy.io
import scipy.linalg
import sklearn.metrics


def kernel(ker, X1, X2, gamma):
    K = None
    if not ker or ker == 'primal':
        K = X1
    elif ker == 'linear':
        if X2 is not None:
            K = sklearn.metrics.pairwise.linear_kernel(np.asarray(X1).T, np.asarray(X2).T)
        else:
            K = sklearn.metrics.pairwise.linear_kernel(np.asarray(X1).T)
    elif ker == 'rbf':
        if X2 is not None:
            K = sklearn.metrics.pairwise.rbf_kernel(np.asarray(X1).T, np.asarray(X2).T, gamma)
        else:
            K = sklearn.metrics.pairwise.rbf_kernel(np.asarray(X1).T, None, gamma)
    return K


class MTCA:
    def __init__(self, kernel_type='primal', dim=30, lamb=1, gamma=1):
        """
        Init func
        :param kernel_type: kernel, values: 'primal' | 'linear' | 'rbf'
        :param dim: dimension after transfer
        :param lamb: lambda value in equation
        :param gamma: kernel bandwidth for rbf kernel
        """
        self.kernel_type = kernel_type
        self.dim = dim
        self.lamb = lamb
        self.gamma = gamma
        self.A = None
        self.X_raw = None

    def fit(self, Xs):
        """
        Transform Xs and Xt
        :param Xs: a list of ns * n_feature source feature
        :return: Xs_new and Xt_new after TCA
        """
        X = np.vstack(Xs).T
        self.X_raw = X
        m, n = X.shape

        ns, nt = len(Xs), len(Xt)
        e = np.vstack((1 / ns * np.ones((ns, 1)), -1 / nt * np.ones((nt, 1))))
        M = e * e.T
        M = M / np.linalg.norm(M, 'fro')
        H = np.eye(n) - 1 / n * np.ones((n, n))
        K = kernel(self.kernel_type, X, X, gamma=self.gamma)
        n_eye = m if self.kernel_type == 'primal' else n
        a, b = np.linalg.multi_dot([K, M, K.T]) + self.lamb * np.eye(n_eye), np.linalg.multi_dot([K, H, K.T])
        w, V = scipy.linalg.eig(a, b)
        ind = np.argsort(w)
        A = V[:, ind[:self.dim]]
        self.A = A
        Z = np.dot(A.T, K)
        Xs_new = Z[:, :ns].T
        return Xs_new

    def transform(self, X):
        X = X.T
        K = kernel(self.kernel_type, X, self.X_raw, gamma=self.gamma)
        K = self.A.T @ K
        return K.T

