from .metrics import get_metrics, accuracy, class_accuracy, get_confusion_matrix
from .losses import LossInfo
import torch
import numpy as np


def test_metrics_add_properly():        
    y_pred = torch.as_tensor([
        [0.01, 0.90, 0.09],
        [0.01, 0, 0.99],
        [0.01, 0, 0.99],
    ])
    y = torch.as_tensor([
        1,
        2,
        0,
    ], dtype=torch.int64)
    m1 = get_metrics(y_pred=y_pred, y=y)
    assert m1.n_samples == 3
    assert np.isclose(m1.accuracy, 2/3)
    
    y_pred = torch.as_tensor([
        [0.01, 0.90, 0.09],
        [0.01, 0, 0.99],
        [0.01, 0, 0.99],
        [0.01, 0, 0.99],
        [0.01, 0, 0.99],
    ])
    y = torch.as_tensor([
        1,
        2,
        2,
        0,
        0,
    ])
    m2 = get_metrics(y_pred=y_pred, y=y)
    assert m2.n_samples == 5
    assert np.isclose(m2.accuracy, 3/5)

    m3 = m1 + m2
    assert m3.n_samples == 8
    assert np.isclose(m3.accuracy, 5/8)

def test_metrics_from_tensors():
    y_pred = torch.as_tensor([
        [0.01, 0.90, 0.09],
        [0.01, 0, 0.99],
        [0.01, 0, 0.99],
    ])
    y = torch.as_tensor([
        1,
        2,
        0,
    ])
    m = get_metrics(y_pred=y_pred, y=y)
    assert m.n_samples == 3
    assert np.isclose(m.accuracy, 2/3)

def test_accuracy():
    y_pred = torch.as_tensor([
        [0.01, 0.90, 0.09],
        [0.01, 0, 0.99],
        [0.01, 0, 0.99],
    ])
    y = torch.as_tensor([
        1,
        2,
        0,
    ])
    assert np.isclose(accuracy(y_pred, y), 2/3)


def test_per_class_accuracy_perfect():
    y_pred = torch.as_tensor([
        [0.1, 0.9, 0.0],
        [0.1, 0.0, 0.9],
        [0.1, 0.4, 0.5],
        [0.9, 0.1, 0.0],
    ])
    y = torch.as_tensor([
        1,
        2,
        2,
        0,
    ])
    expected = [1, 1, 1]
    class_acc = class_accuracy(y_pred, y).numpy().tolist()
    assert class_acc == expected


def test_per_class_accuracy_zero():
    y_pred = torch.as_tensor([
        [0.1, 0.9, 0.0],
        [0.1, 0.9, 0.0],
        [0.1, 0.9, 0.0],
        [0.1, 0.9, 0.0],
    ])
    y = torch.as_tensor([
        0,
        0,
        0,
        0,
    ])
    expected = [0, 0, 0]
    class_acc = class_accuracy(y_pred, y).numpy().tolist()
    assert class_acc == expected


def test_confusion_matrix():
    y_pred = torch.as_tensor([
        [0.1, 0.9, 0.0],
        [0.1, 0.4, 0.5],
        [0.1, 0.9, 0.0],
        [0.9, 0.0, 0.1],
    ])
    y = torch.as_tensor([
        0,
        0,
        1,
        0,
    ])
    expected = [
        [1, 1, 1],
        [0, 1, 0],
        [0, 0, 0],
    ]
    confusion_mat = get_confusion_matrix(y_pred, y).numpy().tolist()
    assert confusion_mat == expected

def test_per_class_accuracy_realistic():
    y_pred = torch.as_tensor([
        [0.9, 0.0, 0.0], # correct for class 0
        [0.1, 0.5, 0.4], # correct for class 1
        [0.1, 0.0, 0.9], # correct for class 2
        [0.1, 0.8, 0.1], # wrong, should be 1
        [0.1, 0.0, 0.9], # wrong, should be 0
        [0.9, 0.0, 0.0], # wrong, should be 1
        [0.1, 0.5, 0.4], # wrong, should be 2
        [0.1, 0.4, 0.5], # correct for class 2
    ])
    y = torch.as_tensor([
        0,
        1,
        2,
        0, 
        0,
        1,
        2,
        2,
    ])
    expected = [1/3, 1/2, 2/3]
    class_acc = class_accuracy(y_pred, y).numpy().tolist()
    assert all(np.isclose(class_acc, expected))