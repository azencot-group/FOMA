import torch
import numpy as np
import random
import pickle

import os
from scipy.spatial.distance import pdist, squareform
from intrinsic_dimension import intrinsic_dimension
from config import dataset_defaults

def get_config(args):
    dict_name = args.dataset
    args_dict = args.__dict__
    args.dataset_name = args.dataset
    args.data_dir = f"data/{dict_name}/"
    dataset_config = dataset_defaults.get(dict_name, {})
    if dict_name.startswith("TimeSeries"):
        dataset_config["dataset"] = "TimeSeries"
        dataset_config["ts_name"] = dict_name
        args.dataset_name = args.dataset.split('-')[1]
    args_dict.update(dataset_config)
    args_dict.update(dataset_config['command_args'])
    return args_dict


########## seed init ##########
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


######### get status information ###########
def stats_values(targets):
    mean = np.mean(targets)
    min = np.min(targets)
    max = np.max(targets)
    std = np.std(targets)
    print(f'y stats: mean = {mean}, max = {max}, min = {min}, std = {std}')
    return mean, min, max, std


######### get file name ###########
def get_unique_file_name(args, extra_str2='', profix='.txt'):
    if args.dataset == 'TimeSeries':
        extra_str = '_' + args.ts_name
    elif args.is_ood == True:
        extra_str = '_OOD'
    else:
        extra_str = ''

    if extra_str2 != '':
        extra_str += '_' + extra_str2

    if args.dataset == 'Dti_dg' and args.sub_sample_batch_max_num != -1:
        extra_str += f'_sub{args.sub_sample_batch_max_num}'

    if args.seed != 0:
        extra_str += '_Seed' + str(args.seed)

    if args.dataset == 'RotateFashionMNIST':
        if args.construct_color_data:
            extra_str += '_Color'
        if args.construct_no_color_data:
            extra_str += '_NoColor'

    if args.batch_selection == 'erm':
        fn = f"{args.dataset}{extra_str}_{args.batch_selection}"
    else:
        fn = f"{args.dataset}{extra_str}_{args.batch_selection}_{'UseManifold' if args.foma_latent else 'NotUseManifold'}"

    fn += profix
    return fn


#### write result and model #####

def write_result(args, data, result_path, extra_str=''):
    full_path = result_path + get_unique_file_name(args, extra_str, '.txt')
    if args.show_process:
        print(f'write result into path: {full_path}')
    with open(full_path, 'a+') as f:  # >>
        # f.write(f'{args.seed}:{r}\n')
        if isinstance(data, list):
            for i in range(len(data)):
                f.write(f'{data[i]}\t')
            f.write(f'\n')
        elif isinstance(data, dict):  # write result dict
            f.write(f'seed = {args.seed}\n')
            for k in data.keys():
                f.write(f'{k}\t\t')
            f.write(f'\n')
            for k in data.keys():
                f.write('{:.7f}\t'.format(data[k]))
            f.write(f'\n')
        else:
            f.write(f'{data}\n')


def write_model(args, model, result_path, extra_str=''):
    if model != None:
        pt_full_path = result_path + get_unique_file_name(args, extra_str, '.pickle')
        if args.show_process:
            print(f'write model into path: {pt_full_path}')
        ##### store best model #####
        s = pickle.dumps(model)
        with open(pt_full_path, 'wb+') as f:
            f.write(s)


#
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
        id_est = intrinsic_dimension(np.concatenate((X, Y), axis=1))
        id_est = int(np.ceil(id_est))
        np.save(id_path, id_est)
    else:
        id_est = int(np.load(id_path))

    return id_est

