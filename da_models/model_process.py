# -*- coding: utf-8 -*-
"""
Created on 2020/11/11 11:06

@author: John_Fengz
"""
import numpy as np
from sklearn.decomposition import PCA
from da_models.tca import TCA
from da_models.coral import CORAL
from da_models.none_mdas import MDAS
from utils.data_utils import normalize
from utils.sampling import sampling_ds, sampling_ns, random_sampling_ds
from sklearn.decomposition import MiniBatchDictionaryLearning


def non_process(train, test):

    x_tar_ns, x_sou_ns, x_sou_ds, y_train = train
    X_train = np.vstack(x_sou_ds)
    y_train = np.vstack(y_train)[:, 0]

    X_test, y_test = test
    y_test = y_test[:, 0]

    X_train, X_test = normalize(X_train, X_test)
    return X_train, X_test, y_train, y_test


def sampling_process(train, test, random_state=42):
    x_tar_ns, x_sou_ns, x_sou_ds, y_train = train
    x_sou_ds_new, y_train_new = random_sampling_ds(x_sou_ds, y_train, random_state=random_state)
    X_train = np.vstack(x_sou_ds_new)
    y_train = np.vstack(y_train_new)[:, 0]

    X_test, y_test = test
    y_test = y_test[:, 0]

    X_train, X_test = normalize(X_train, X_test)
    return X_train, X_test, y_train, y_test


def tca_process(train, test, p=50):
    x_tar_ns, x_sou_ns, x_sou_ds, y_train = train
    y_train = np.vstack(y_train)[:, 0]
    x_tar_ds, y_test = test
    x_sou_ns_mat = np.vstack(x_sou_ns)
    x_sou_ds_mat = np.vstack(x_sou_ds)

    model = TCA(dim=p)
    model.fit(x_sou_ns_mat, x_tar_ns)
    X_train = model.transform(x_sou_ds_mat)
    X_test = model.transform(x_tar_ds)

    X_train, X_test = normalize(X_train, X_test)
    return X_train, X_test, y_train, y_test[:, 0]


def coral_process(train, test):
    x_tar_ns, x_sou_ns, x_sou_ds, y_train = train
    y_train = np.vstack(y_train)[:, 0]
    x_tar_ds, y_test = test
    x_sou_mat = np.vstack(x_sou_ns + x_sou_ds)
    len_x_sou_ns = len(np.vstack(x_sou_ds))

    model = CORAL()
    model.fit(x_sou_mat, x_tar_ns)
    X_train = model.transform(x_sou_mat)
    X_test = model.transform(x_tar_ds)

    X_train, X_test = normalize(X_train, X_test)
    return X_train[-len_x_sou_ns:], X_test, y_train, y_test[:, 0]


def pca_process(train, test):
    x_tar_ns, x_sou_ns, x_sou_ds, y_train = train
    y_train = np.vstack(y_train)[:, 0]
    x_tar_ds, y_test = test
    x_sou_mat = np.vstack(x_sou_ds)

    model = PCA(n_components=50)
    X_train = model.fit_transform(x_sou_mat)
    X_test = model.transform(x_tar_ds)

    X_train, X_test = normalize(X_train, X_test)
    return X_train, X_test, y_train, y_test[:, 0]


def mdas_process(train, test,
                 n_components=500,
                 p=50,
                 alpha=0.5,
                 lamb1=0.1,
                 lamb2=0.1,
                 random_state=42):
    x_tar_ns, x_sou_ns, x_sou_ds, y_train = train
    x_tar_ds, y_test = test
    x_sou_ds, y_train = sampling_ds(x_sou_ds, y_train, random_state)
    x_sou_ns = sampling_ns(x_tar_ns, x_sou_ns, random_state)
    y_train = np.vstack(y_train)[:, 0]

    # dictionary learning
    dic = x_sou_ns + x_sou_ds
    dic.append(x_tar_ns)
    dic = np.vstack(dic)
    dic = np.unique(dic, axis=0)
    mdl = MiniBatchDictionaryLearning(n_components=n_components, n_iter=150, random_state=42)
    mdl.fit(dic)
    dic = mdl.components_
    print('Dictionary loaded')

    model = MDAS(p=p, alpha=alpha, lamb1=lamb1, lamb2=lamb2, dic=dic)
    x_sou_ds = model.fit(x_tar_ns, x_sou_ns, x_sou_ds)
    X_test = model.transform(x_tar_ds)
    X_train = np.vstack(x_sou_ds)

    X_train, X_test = normalize(X_train, X_test)
    return X_train, X_test, y_train, y_test[:, 0]
