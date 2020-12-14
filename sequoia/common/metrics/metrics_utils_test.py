import torch
import numpy as np

from .metrics_utils import accuracy, class_accuracy, get_confusion_matrix


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
    class_acc = class_accuracy(y_pred, y).tolist()
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
    class_acc = class_accuracy(y_pred, y).tolist()
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
    confusion_mat = get_confusion_matrix(y_pred=y_pred, y=y).tolist()
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
    class_acc = class_accuracy(y_pred, y).tolist()
    assert all(np.isclose(class_acc, expected))
