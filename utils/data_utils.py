# -*- coding: utf-8 -*-
"""
Created on 2020/11/30 11:02

@author: John_Fengz
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def normalize(X_train, X_test):
    scale = MinMaxScaler()
    scale.fit(X_train)
    X_train = scale.transform(X_train)
    X_test = scale.transform(X_test)
    return X_train, X_test


def get_data(file_path, fop):
    data = pd.read_excel(file_path)
    feature = np.array(data.iloc[:, :70])
    feature /= np.linalg.norm(feature, axis=0)
    feature_ns = feature[:fop]
    feature_ds = feature[fop:]
    rul_ds = np.array(data.iloc[:, -1])[fop:]
    rul_ds = rul_ds[:, np.newaxis]
    ms = MinMaxScaler()
    rul_ds = ms.fit_transform(rul_ds)
    return feature_ns, feature_ds, rul_ds


def train_test_split(feature_ns, feature_ds, rul_ds, target_num):
    if target_num > len(feature_ns) - 1:
        raise ValueError('The target_num should not be more than the number of domains')
    train_feature_ns = feature_ns[:target_num] + feature_ns[(target_num + 1):]
    train_feature_ds = feature_ds[:target_num] + feature_ds[(target_num + 1):]
    train_rul_ds = rul_ds[:target_num] + rul_ds[(target_num + 1):]
    train = (feature_ns[target_num], train_feature_ns, train_feature_ds,
             train_rul_ds)
    test = (feature_ds[target_num], rul_ds[target_num])
    return train, test
