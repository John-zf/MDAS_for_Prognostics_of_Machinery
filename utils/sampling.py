# -*- coding: utf-8 -*-
"""
Created on 2020/11/30 11:03

@author: John_Fengz
"""
import numpy as np
from sklearn.utils import check_random_state, _safe_indexing


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


def random_sampling_ns(x_tar_ns, x_sou_ns, random_state):
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
            indices = random_state.randint(0, len(indices), num_samples)
            indices = np.sort(indices)
            x_sou_ns_new.append(_safe_indexing(x, indices))
    return x_sou_ns_new


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
            indices = np.sort(indices)
            x_sou_ds_new.append(_safe_indexing(x, indices))
            y_train_new.append(_safe_indexing(y, indices))
        elif len(x) < num_samples:
            # over sampling
            indices = random_state.randint(0, len(indices), num_samples)
            indices = np.sort(indices)
            x_sou_ds_new.append(_safe_indexing(x, indices))
            y_train_new.append(_safe_indexing(y, indices))
    return x_sou_ds_new, y_train_new
