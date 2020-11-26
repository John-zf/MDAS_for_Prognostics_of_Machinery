# -*- coding: utf-8 -*-
"""
Created on 2020/11/26 9:39

@author: John_Fengz
"""

import numpy as np
from sklearn import svm
from sklearn.metrics import mean_absolute_error
from da_models import model_process
from da_models.utils import rmse_score, get_data, train_test_split


def base_model():
    return svm.SVR(kernel='rbf')


# load datasets
folder = '../data/'
feature_ns1, feature_ds1, rul_ds1 = get_data(folder + 'xBearing1_1.xlsx', 1490)
feature_ns2, feature_ds2, rul_ds2 = get_data(folder + 'xBearing1_2.xlsx', 827)
feature_ns3, feature_ds3, rul_ds3 = get_data(folder + 'xBearing1_3.xlsx', 1684)
feature_ns4, feature_ds4, rul_ds4 = get_data(folder + 'xBearing1_4.xlsx', 1083)
feature_ns5, feature_ds5, rul_ds5 = get_data(folder + 'xBearing1_5.xlsx', 680)
feature_ns6, feature_ds6, rul_ds6 = get_data(folder + 'xBearing1_6.xlsx', 649)
feature_ns7, feature_ds7, rul_ds7 = get_data(folder + 'xBearing1_7.xlsx', 1026)
print('Data loaded')

# train\test split
feature_ns = [feature_ns1, feature_ns2, feature_ns3, feature_ns4, feature_ns5,
              feature_ns6, feature_ns7]
feature_ds = [feature_ds1, feature_ds2, feature_ds3, feature_ds4, feature_ds5,
              feature_ds6, feature_ds7]
rul_ds = [rul_ds1, rul_ds2, rul_ds3, rul_ds4, rul_ds5, rul_ds6, rul_ds7]
train, test = train_test_split(feature_ns, feature_ds, rul_ds, 6)

# MDAS

X_train, X_test, y_train, y_test = model_process.mdas_process(train, test)
regr = base_model()
regr.fit(X_train, y_train)
y_pred = np.clip(regr.predict(X_test), 0, 1)
print('rmse of MDAS:{}'.format(rmse_score(y_test, y_pred)))
print('mae of MDAS:{}'.format(mean_absolute_error(y_test, y_pred)))
