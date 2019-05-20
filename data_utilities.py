import torch
import numpy as np
from collections import Iterable


def features_labels_split(data):
    """ how to manage `dtype` of `data` better? """
    assert isinstance(data, Iterable), \
        'data must be an array of sample. i.e. two-tuples (feature, label).'
    feature_s, label_s = [], []
    for datum in data:
        assert len(datum) == 2, \
            'data must be an array of sample. i.e. two-tuples (feature, label).'
        feature, label = datum
        feature_s.append(feature)
        label_s.append(label)
    if isinstance(feature_s[0], np.ndarray):
        return np.array(feature_s), np.array(label_s)
    elif isinstance(feature_s[0], torch.Tensor):
        label_s = torch.stack(label_s, dim=0) if isinstance(label_s[0], torch.Tensor) \
            else torch.tensor(label_s)
        return torch.stack(feature_s, dim=0), label_s
    else:
        return feature_s, label_s


def one_hot_encode(label_s, reverse=False):
    assert isinstance(label_s, Iterable), 'labels should be an list (array-like) of label.'
    if not reverse:
        """ [todo] general type-casting? """
        label_s_ = np.array(label_s)
        # import pdb
        # pdb.set_trace()
        """ note the dtype of arr returned by `np.zeros`. """
        label_one_hot_s = np.zeros((label_s_.size, label_s_.max() + 1), dtype=np.int)
        label_one_hot_s[np.arange(label_s_.size), label_s_] = 1
        if isinstance(label_s, np.ndarray):
            return label_one_hot_s
        elif isinstance(label_s, torch.Tensor):
            return torch.from_numpy(label_one_hot_s)
        else:
            return list(label_one_hot_s)
    else:
        if isinstance(label_s, np.ndarray):
            return np.argmax(label_s, 1)
        elif isinstance(label_s, torch.Tensor):
            return torch.argmax(label_s, 1)
        else:
            return list(np.argmax(label_s, 1))
