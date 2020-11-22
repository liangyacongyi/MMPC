#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   utils.py
@Contact :   liangyacongyi@163.com

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2020/1/2 09:10 AM   liangcong    1.0
"""
import tensorflow as tf
import numpy as np
from numpy import *


def sim_dis_tensorflow(x, y):
    """
    calculate the distance of two matrices along the second dimension using tensorflow
    :param x: matrix x
    :param y: matrix y
    :return: matrix D
    """
    y_G = tf.matmul(y, tf.transpose(y, [1, 0]))
    y_H = tf.tile(tf.expand_dims(tf.diag_part(y_G), 0), (tf.shape(x)[0], 1))
    x_G = tf.matmul(x, tf.transpose(x, [1, 0]))
    x_H = tf.tile(tf.expand_dims(tf.diag_part(x_G), 0), (tf.shape(y)[0], 1))
    R = tf.matmul(x, tf.transpose(y, [1, 0]))
    D = y_H + tf.transpose(x_H, [1, 0]) - 2*R
    return 0.5*D


def _zca(X, U_matrix=None, S_matrix=None, mu=None, flag='train', alpha=1e-5):
    """
    preprocess image dataset by zca
    :param X: image dataset, format: n*d, n: total samples, d: dimentions
    :return:
    """
    if flag == 'train':
        # mu = np.mean(X, axis=0)
        # X = X - mu
        cov = np.cov(X.T)
        U, S, V = np.linalg.svd(cov)
    else:
        # X = X - mu
        U = U_matrix
        S = S_matrix
    x_rot = np.dot(X, U)
    pca_whiten = x_rot / np.sqrt(S + alpha)
    zca_whiten = np.dot(pca_whiten, U.T)
    return zca_whiten, U, S, mu


def _gcn(X, flag=0, scale=55.):
    """
    preprocess image dataset by gcn
    :param X: image dataset, format: n*d, n: total samples, d: dimentions
    :return: the dataset after preprocessing by gcn
    """
    if flag == 0:
        print("1")
        mean = np.mean(X, axis=1)
        X = X - mean[:, np.newaxis]
        contrast = np.sqrt(10. + (X**2).sum(axis=1)) / scale
        contrast[contrast < 1e-8] = 1.
        X = X / contrast[:, np.newaxis]
    else:
        print("3")
        X = X.reshape([-1, 32, 32, 3])
        mu = np.mean(X, axis=(1, 2)).reshape([-1, 1, 1, 3])
        std = np.std(X, axis=(1, 2)).reshape([-1, 1, 1, 3])
        X = X - mu
        X = X / std
        X = X.reshape([-1, 3072])
    return X


def MMPC_loss(output, label, k, flag):
    """
    calculate unbiased constrained loss based on given top k
    :param output: final output of CNN models
    :param label: ground truth one-hot coding label
    :param k: total number of category
    :param flag: 1 means basic version, 2 means exp() version, 3 means neg log() version
    :return: unbiased constrained loss
    """
    a = tf.multiply(output, 1-label)
    _a, _ = tf.nn.top_k(a, k=k)
    if flag == 2:
        _a = tf.exp(_a) - 1
    elif flag == 3:
        _a = -1 * tf.log(1 - _a)
    else:
        _a = _a
    _MMPC_loss = tf.reduce_mean(_a)
    return _MMPC_loss

