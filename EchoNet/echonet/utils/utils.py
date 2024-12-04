import torch
import numpy as np
import random
import pickle

import os.path
import pandas as pd

from filelock import FileLock
from intrinsic_dimension import intrinsic_dimension_gpu
from scipy.spatial.distance import pdist, squareform



########## seed init ##########
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_dist_matrix(data):
    distances = pdist(data)
    dist_matrix = squareform(distances)
    return dist_matrix


def normalize_probabilities(dist_matrix):
    for row in dist_matrix:
        # Find the smallest non-zero value
        smallest_non_zero = np.min(row[np.nonzero(row)])
        # Replace zero values with the smallest non-zero value
        row[row == 0] = smallest_non_zero
    probabilities = 1.0 / np.where(dist_matrix != 0, dist_matrix, 1e-8)
    np.fill_diagonal(probabilities, 0)
    probabilities /= np.sum(probabilities, axis=1, keepdims=True)
    return probabilities


def get_probabilities(args, probabilities_dir, Y):
    f_name = f'{args.dataset_name}_{args.batch_selection}.npy'
    f_path = os.path.join(probabilities_dir, f_name)
    probabilities = None
    if not os.path.exists(f_path) and (args.batch_selection == 'knn' or args.batch_selection == 'knnp'):
        if Y.ndim < 2:
            Y = Y.unsqueeze(1)
        dist_matrix = get_dist_matrix(Y)

    if args.batch_selection == 'knn':
        if not os.path.exists(f_path):
            np.fill_diagonal(dist_matrix, -np.inf)
            probabilities = np.argsort(dist_matrix, axis=1)
            np.save(f_path, probabilities)
        else:
            probabilities = np.load(f_path)
    elif args.batch_selection == 'knnp':
        if not os.path.exists(f_path):
            probabilities = normalize_probabilities(dist_matrix)
            np.save(f_path, probabilities)
        else:
            probabilities = np.load(f_path)
    return probabilities


def get_id(id_path, X, Y):
    if not os.path.exists(id_path):
        X = X.reshape(X.shape[0], -1)
        id_est = intrinsic_dimension_gpu(np.concatenate((X, Y), axis=1))
        id_est = int(np.ceil(id_est))
        np.save(id_path, id_est)
    else:
        id_est = int(np.load(id_path))

    return id_est
