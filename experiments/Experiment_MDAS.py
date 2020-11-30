# -*- coding: utf-8 -*-
"""
Created on 2020/11/26 9:39

@author: John_Fengz
"""

import numpy as np
from sklearn import svm
from sklearn.metrics import mean_absolute_error
from da_models import model_process
from utils.data_utils import get_data, train_test_split
from utils.metrics import rmse_score
import logging
import time


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
log_name = 'Tuning-MDAS-{}.log'.format(time.strftime('%Y-%m-%d-%H-%M'))
log_path = '../logs/' + log_name
formatter = logging.Formatter('[%(asctime)s]    %(message)s')
logger = logging.getLogger('MDAS')
logger.setLevel(logging.DEBUG)

file = logging.FileHandler(log_path)
file.setLevel(logging.DEBUG)
file.setFormatter(formatter)

stream = logging.StreamHandler()
stream.setLevel(logging.DEBUG)
stream.setFormatter(formatter)

logger.addHandler(stream)
logger.addHandler(file)

logger.info('   '.join(['para_count', 'n_components', 'p', 'alpha', 'lamb1', 'lamb2', 'random_state', 'rmse', 'mae']))

seed = 42
n_components_all = [600, 500, 400, 300, 200, 100]
p_all = [70, 60, 50, 40, 30]
alpha_all = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1]
lamb1_all = [0.0001, 0.001, 0.01, 0.1, 1, 10]
lamb2_all = [0.0001, 0.001, 0.01, 0.1, 1, 10]

count = 0
for n_components in n_components_all:
    for p in p_all:
        for alpha in alpha_all:
            for lamb1 in lamb1_all:
                for lamb2 in lamb2_all:
                    count += 1
                    for i in range(20):
                        print(i)
                        random_state = seed + i
                        X_train, X_test, y_train, y_test = \
                            model_process.mdas_process(train, test,
                                                       n_components=n_components,
                                                       p=p,
                                                       alpha=alpha,
                                                       lamb1=lamb1,
                                                       lamb2=lamb2,
                                                       random_state=random_state)
                        regr = base_model()
                        regr.fit(X_train, y_train)
                        y_pred = np.clip(regr.predict(X_test), 0, 1)
                        rmse = rmse_score(y_test, y_pred)
                        mae = mean_absolute_error(y_test, y_pred)
                        logger.info('   '.join([str(count), str(n_components), str(p),
                                                str(alpha), str(lamb1), str(lamb2), str(random_state),
                                                str(rmse), str(mae)]))
