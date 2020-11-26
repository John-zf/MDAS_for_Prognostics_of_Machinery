# -*- coding: utf-8 -*-
"""
Sun B, Feng J, Saenko K.
Return of frustratingly easy domain adaptation[C]
AAAI. 2016, 6(7): 8.
"""

import numpy as np
import scipy.linalg


class CORAL:
    def __init__(self):
        self.A = None

    def fit(self, Xs, Xt):
        """
        Perform CORAL on the source domain features
        :param Xs: ns * n_feature, source feature
        :param Xt: nt * n_feature, target feature
        :return: New source domain features
        """
        cov_src = np.cov(Xs.T) + np.eye(Xs.shape[1])
        cov_tar = np.cov(Xt.T) + np.eye(Xt.shape[1])
        A_coral = np.dot(scipy.linalg.fractional_matrix_power(cov_src, -0.5),
                         scipy.linalg.fractional_matrix_power(cov_tar, 0.5))
        self.A = A_coral
        Xs_new = np.real(np.dot(Xs, self.A))
        return Xs_new

    def transform(self, X):
        X_t = X @ self.A
        return X_t

