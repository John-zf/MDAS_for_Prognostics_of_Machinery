# -*- coding: utf-8 -*-
"""
Created on 2020/11/11 11:18

@author: John_Fengz
"""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import check_random_state, _safe_indexing


def rmse_score(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


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


def sampling_ns(x_tar_ns, x_sou_ns, random_state):
    random_state = check_random_state(random_state)
    x_sou_ns_new = []
    num_samples = len(x_tar_ns)
    for x in x_sou_ns:
        indices = np.array(range(x.shape[0]))
        if len(x) == num_samples:
            x_sou_ns_new.append(x)
        elif len(x) > num_samples:
            # under sampling
            indices = random_state.choice(indices, num_samples)
            indices = np.sort(indices)
            x_sou_ns_new.append(_safe_indexing(x, indices))
        elif len(x) < num_samples:
            # over sampling
            sample_indices = random_state.randint(0, len(indices), num_samples-len(x))
            indices = np.append(indices, sample_indices)
            indices = np.sort(indices)
            x_sou_ns_new.append(_safe_indexing(x, indices))
    return x_sou_ns_new


def sampling_ds(x_sou_ds, y_train, random_state):
    random_state = check_random_state(random_state)
    x_sou_ds_new = []
    y_train_new = []
    num_samples = int(np.mean([len(x) for x in x_sou_ds]))
    for (x, y) in zip(x_sou_ds, y_train):
        indices = np.array(range(x.shape[0]))
        if len(x) == num_samples:
            x_sou_ds_new.append(x)
            y_train_new.append(y)
        elif len(x) > num_samples:
            # under sampling
            indices = random_state.choice(indices, num_samples)
            indices = np.sort(indices)
            x_sou_ds_new.append(_safe_indexing(x, indices))
            y_train_new.append(_safe_indexing(y, indices))
        elif len(x) < num_samples:
            # over sampling
            sample_indices = random_state.randint(0, len(indices), num_samples - len(x))
            indices = np.append(indices, sample_indices)
            indices = np.sort(indices)
            x_sou_ds_new.append(_safe_indexing(x, indices))
            y_train_new.append(_safe_indexing(y, indices))
    return x_sou_ds_new, y_train_new


def over_sampling(samples, sample_indices, random_state):
    rand_sys = random_state.rand(len(sample_indices))
    new_samples = []
    for idx, i in enumerate(sample_indices):
        sys_sample = samples[i] + rand_sys[idx] * (2 * samples[i] - samples[i - 1] - samples[i + 1])
        new_samples.append(sys_sample)
    return np.array(new_samples)


def random_sampling_ds(x_sou_ds, y_train, random_state=30):
    random_state = check_random_state(random_state)
    x_sou_ds_new = []
    y_train_new = []
    num_samples = int(np.mean([len(x) for x in x_sou_ds]))
    for (x, y) in zip(x_sou_ds, y_train):
        indices = np.array(range(x.shape[0]))
        if len(x) == num_samples:
            x_sou_ds_new.append(x)
            y_train_new.append(y)
        elif len(x) > num_samples:
            # under sampling
            indices = random_state.choice(indices, num_samples)
            x_sou_ds_new.append(_safe_indexing(x, indices))
            y_train_new.append(_safe_indexing(y, indices))
        elif len(x) < num_samples:
            # over sampling
            sample_indices = random_state.randint(1, len(indices)-1, num_samples - len(x))
            new_samples = over_sampling(x, sample_indices, random_state)
            x_sou_ds_new.append(np.vstack((x, new_samples)))

            indices = np.append(indices, sample_indices)
            y_train_new.append(_safe_indexing(y, indices))
    return x_sou_ds_new, y_train_new
