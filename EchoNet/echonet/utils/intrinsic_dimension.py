#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: alessio ansuini (alessioansuini@gmail.com)
"""

from math import floor
import numpy as np
import torch
from scipy.stats import pearsonr
import math
from sklearn import linear_model
from scipy.spatial.distance import pdist, squareform

def intrinsic_dimension(X, verbose=False):
    """
    Estimates the intrinsic dimension of a dataset.

    Args:
        X (array-like): Input data matrix.
        verbose (bool): If True, prints additional information.

    Returns:
        float: Estimated intrinsic dimension.
    """
    p_dist = pdist(X, metric='euclidean')
    dist = squareform(p_dist)
    est_np = estimate(dist, verbose=verbose)
    return est_np


def intrinsic_dimension_torch(X, verbose=False):
    if not torch.is_tensor(X):
        X = torch.tensor(X, device='cuda')
    elif not X.is_cuda:
        X = torch.tensor(X, device='cuda')

    try:
        est = estimate_torch(torch.cdist(X, X, p=2, compute_mode='donot_use_mm_for_euclid_dist'), verbose=verbose)
    except RuntimeError:
        print('Full distances estimation failed, trying batched estimation')
        est = estimate_torch_batched(X, verbose=verbose)
    return est


def batched_cdist(data, batch_size=1000):
    n = data.shape[0]
    dist_matrix = torch.zeros((n, n), device='cpu')  # Create the distance matrix on GPU

    for i in range(0, n, batch_size):
        end_i = min(i + batch_size, n)

        # Convert the current batch to a tensor and move to GPU
        batch_i = torch.tensor(data[i:end_i], dtype=torch.float32).cuda()

        # Compute distances for the current batch to all points in the full dataset
        for j in range(0, n, batch_size):
            end_j = min(j + batch_size, n)

            # Convert the second batch to a tensor and move to GPU
            batch_j = torch.tensor(data[j:end_j], dtype=torch.float32).cuda()

            # Compute the pairwise distances
            dists = torch.cdist(batch_i, batch_j, p=2, compute_mode='donot_use_mm_for_euclid_dist')

            # Store the results in the distance matrix
            dist_matrix[i:end_i, j:end_j] = dists

    return dist_matrix


def estimate(X, fraction=0.9, verbose=False):
    '''
        Estimates the intrinsic dimension of a system of points from
        the matrix of their distances X

        Args:
        X : 2-D Matrix X (n,n) where n is the number of points
        fraction : fraction of the data considered for the dimensionality
        estimation (default : fraction = 0.9)

        Returns:
        x : log(mu)    (*)
        y : -(1-F(mu)) (*)
        reg : linear regression y ~ x structure obtained with scipy.stats.linregress
        (reg.slope is the intrinsic dimension estimate)
        r : determination coefficient of y ~ x
        pval : p-value of y ~ x

        (*) See cited paper for description

        Usage:

        _,_,reg,r,pval = estimate(X,fraction=0.85)

        The technique is described in :

        "Estimating the intrinsic dimension of datasets by a
        minimal neighborhood information"
        Authors : Elena Facco, Maria d’Errico, Alex Rodriguez & Alessandro Laio
        Scientific Reports 7, Article number: 12140 (2017)
        doi:10.1038/s41598-017-11873-y

    '''

    # sort distance matrix
    Y = np.sort(X, axis=1, kind='quicksort')

    # clean data
    k1 = Y[:, 1]
    k2 = Y[:, 2]

    zeros = np.where(k1 == 0)[0]
    if verbose:
        print('Found n. {} elements for which r1 = 0'.format(zeros.shape[0]))
        print(zeros)

    degeneracies = np.where(k1 == k2)[0]
    if verbose:
        print('Found n. {} elements for which r1 = r2'.format(degeneracies.shape[0]))
        print(degeneracies)

    good = np.setdiff1d(np.arange(Y.shape[0]), np.array(zeros))
    good = np.setdiff1d(good, np.array(degeneracies))

    if verbose:
        print('Fraction good points: {}'.format(good.shape[0] / Y.shape[0]))

    k1 = k1[good]
    k2 = k2[good]

    # n.of points to consider for the linear regression
    npoints = int(np.floor(good.shape[0] * fraction))

    # define mu and Femp
    N = good.shape[0]
    mu = np.sort(np.divide(k2, k1), axis=None, kind='quicksort')
    Femp = (np.arange(1, N + 1, dtype=np.float64)) / N

    # take logs (leave out the last element because 1-Femp is zero there)
    x = np.log(mu[:-2])
    y = -np.log(1 - Femp[:-2])

    # regression
    regr = linear_model.LinearRegression(fit_intercept=False)
    regr.fit(x[0:npoints, np.newaxis], y[0:npoints, np.newaxis])
    # r, pval = pearsonr(x[0:npoints], y[0:npoints])
    return regr.coef_[0][0]


def estimate_torch_batched(X, batch_size=2**11, fraction=0.9, verbose=False):
    '''
        Estimates the intrinsic dimension of a system of points from
        the matrix of their distances X

        Args:
        X : 2-D Matrix X (n,d) where n is the number of points and d is the number of dimensions
        fraction : fraction of the data considered for the dimensionality
        estimation (default : fraction = 0.9)

        Returns:
        x : log(mu)    (*)
        y : -(1-F(mu)) (*)
        reg : linear regression y ~ x structure obtained with scipy.stats.linregress
        (reg.slope is the intrinsic dimension estimate)
        r : determination coefficient of y ~ x
        pval : p-value of y ~ x

        (*) See cited paper for description

        Usage:

        _,_,reg,r,pval = estimate(X,fraction=0.85)

        The technique is described in :

        "Estimating the intrinsic dimension of datasets by a
        minimal neighborhood information"
        Authors : Elena Facco, Maria d’Errico, Alex Rodriguez & Alessandro Laio
        Scientific Reports 7, Article number: 12140 (2017)
        doi:10.1038/s41598-017-11873-y

    '''  # sort distance matrix
    X_torch = X
    k1_torch = torch.empty(X_torch.shape[0], device='cuda')
    k2_torch = torch.empty(X_torch.shape[0], device='cuda')

    for ii in range(math.ceil(X_torch.shape[0] / batch_size)):
        X_ii = X_torch[ii * batch_size: (ii + 1) * batch_size]
        c_dist_ii = torch.cdist(X_ii, X_torch, p=2, compute_mode='donot_use_mm_for_euclid_dist')
        X_ii_topk, _ = torch.topk(c_dist_ii, 3, dim=1, largest=False)
        k1_torch[ii * batch_size: (ii + 1) * batch_size] = X_ii_topk[:, 1]
        k2_torch[ii * batch_size: (ii + 1) * batch_size] = X_ii_topk[:, 2]
        if verbose:
            print(f'ii = {ii} / {math.ceil(X_torch.shape[0] / batch_size)}')

    # clean data

####
    k1 = k1_torch.detach().cpu().numpy()
    k2 = k2_torch.detach().cpu().numpy()
    zeros = np.where(k1 == 0)[0]
    if verbose:
        print('Found n. {} elements for which r1 = 0'.format(zeros.shape[0]))
        print(zeros)

    degeneracies = np.where(k1 == k2)[0]
    if verbose:
        print('Found n. {} elements for which r1 = r2'.format(degeneracies.shape[0]))
        print(degeneracies)

    good = np.setdiff1d(np.arange(X_torch.shape[0]), np.array(zeros))
    good = np.setdiff1d(good, np.array(degeneracies))

    if verbose:
        print('Fraction good points: {}'.format(good.shape[0] / X_torch.shape[0]))

    k1 = k1[good]
    k2 = k2[good]

    # n.of points to consider for the linear regression
    npoints = int(np.floor(good.shape[0] * fraction))

    # define mu and Femp
    N = good.shape[0]
    mu = np.sort(np.divide(k2, k1), axis=None, kind='quicksort')
    Femp = (np.arange(1, N + 1, dtype=np.float64)) / N

    # take logs (leave out the last element because 1-Femp is zero there)
    x = np.log(mu[:-2])
    y = -np.log(1 - Femp[:-2])

    # regression
    regr = linear_model.LinearRegression(fit_intercept=False)
    regr.fit(x[0:npoints, np.newaxis], y[0:npoints, np.newaxis])
    return regr.coef_[0][0]
    ####


def estimate_torch(X, fraction=0.9, verbose=False):
    '''
        Estimates the intrinsic dimension of a system of points from
        the matrix of their distances X

        Args:
        X : 2-D Matrix X (n,n) where n is the number of points
        fraction : fraction of the data considered for the dimensionality
        estimation (default : fraction = 0.9)

        Returns:
        x : log(mu)    (*)
        y : -(1-F(mu)) (*)
        reg : linear regression y ~ x structure obtained with scipy.stats.linregress
        (reg.slope is the intrinsic dimension estimate)
        r : determination coefficient of y ~ x
        pval : p-value of y ~ x

        (*) See cited paper for description

        Usage:

        _,_,reg,r,pval = estimate(X,fraction=0.85)

        The technique is described in :

        "Estimating the intrinsic dimension of datasets by a
        minimal neighborhood information"
        Authors : Elena Facco, Maria d’Errico, Alex Rodriguez & Alessandro Laio
        Scientific Reports 7, Article number: 12140 (2017)
        doi:10.1038/s41598-017-11873-y

    '''  # sort distance matrix
    X_torch = X
    Y_torch, _ = torch.sort(X_torch, dim=1)

    # clean data

    k1_torch = Y_torch[:, 1]
    k2_torch = Y_torch[:, 2]

    zeros_torch = torch.where(k1_torch == 0)[0]
    if verbose:
        print('Found n. {} elements for which r1 = 0'.format(zeros_torch.shape[0]))
        print(zeros_torch)

    degeneracies_torch = torch.where(k1_torch == k2_torch)[0]
    if verbose:
        print('Found n. {} elements for which r1 = r2'.format(degeneracies_torch.shape[0]))
        print(degeneracies_torch)

    Y_range = torch.arange(Y_torch.shape[0], device=X_torch.device)
    good_torch = Y_range[(Y_range[:, None] != zeros_torch).all(dim=1)]

    good_torch = good_torch[(good_torch[:, None] != degeneracies_torch).all(dim=1)]

    if verbose:
        print('Fraction good points: {}'.format(good_torch.shape[0] / Y_torch.shape[0]))

    k1_torch = k1_torch[good_torch]
    k2_torch = k2_torch[good_torch]

    # n.of points to consider for the linear regression
    npoints_torch = int(floor(good_torch.shape[0] * fraction))

    # define mu and Femp
    N_torch = good_torch.shape[0]
    mu_torch, _ = torch.sort(torch.divide(k2_torch, k1_torch), dim=0)
    Femp_torch = (torch.arange(1, N_torch + 1, dtype=X_torch.dtype, device=X_torch.device)) / N_torch

    # take logs (leave out the last element because 1-Femp is zero there)
    x_torch = torch.log(mu_torch[:-2])
    y_torch = -torch.log(1 - Femp_torch[:-2])

    # regression

    coef = x_torch[0:npoints_torch, None].pinverse() @ y_torch[0:npoints_torch, None]
    return coef.item()
