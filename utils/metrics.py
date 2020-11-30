# -*- coding: utf-8 -*-
"""
Created on 2020/11/30 10:56

@author: John_Fengz
"""

import numpy as np
from sklearn.metrics import mean_squared_error


def rmse_score(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))
