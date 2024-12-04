import numpy as np
import copy

import torch
import torch.nn as nn
import time
from torch.optim import Adam
from foma import get_batch_foma


def cal_worst_acc(args, data_packet, best_model_rmse, best_result_dict_rmse, all_begin, device):
    """Calculates the worst accuracy (RMSE or correlation) for out-of-distribution (OOD) datasets."""
    # Check if it's an out-of-distribution (OOD) task
    if not args.is_ood:
        return

    # Extract test data for OOD groups
    x_test_assay_list = data_packet['x_test_assay_list']
    y_test_assay_list = data_packet['y_test_assay_list']

    # Initialize worst accuracy based on the selected metric
    worst_acc = 0.0 if args.metrics == 'rmse' else 1e10

    # Iterate over test assays and evaluate worst accuracy
    for x_test, y_test in zip(x_test_assay_list, y_test_assay_list):
        result_dict = test(args, best_model_rmse, x_test, y_test, '', False, all_begin, device)
        acc = result_dict[args.metrics]

        # Update worst accuracy based on the selected metric
        worst_acc = update_worst_acc(acc, worst_acc, args.metrics)

    # Print and store the worst accuracy in the result dictionary
    print(f'worst {args.metrics} = {worst_acc:.3f}')
    best_result_dict_rmse[f'worst_{args.metrics}'] = worst_acc


def update_worst_acc(current_acc, worst_acc, metric):
    """Updates the worst accuracy based on the metric type (rmse or correlation)."""
    if metric == 'rmse':
        return max(current_acc, worst_acc)
    else:  # For correlation or other metrics
        return min(current_acc, worst_acc, key=abs)


def test(args, model, x_list, y_list, name, need_verbose, epoch_start_time, device):
    model.eval()
    with torch.no_grad():
        # Set batch size and iteration count based on dataset type
        if args.dataset == 'Dti_dg':
            val_iter = x_list.shape[0] // args.batch_size
            val_len = args.batch_size
            y_list = y_list[:val_iter * val_len]
        else:
            val_iter = 1
            val_len = x_list.shape[0]

        y_list_pred = []
        assert val_iter >= 1, "Validation iteration count must be >= 1"

        # Loop through validation iterations
        for ith in range(val_iter):
            # Convert x_list to tensor if it's an ndarray
            x_batch = x_list[ith * val_len:(ith + 1) * val_len]
            if isinstance(x_list, np.ndarray):
                x_batch_torch = torch.tensor(x_batch, dtype=torch.float32).to(device)
            else:
                x_batch_torch = x_batch.to(device)

            # Move model to the appropriate device and get predictions
            model = model.to(device)
            pred_y = model(x_batch_torch).cpu().numpy()
            y_list_pred.append(pred_y)

        # Concatenate predictions and squeeze outputs
        y_list_pred = np.concatenate(y_list_pred, axis=0).squeeze()
        y_list = y_list.squeeze()

        # Convert y_list to numpy if it's a tensor
        if not isinstance(y_list, np.ndarray):
            y_list = y_list.numpy()

        # Calculate metrics
        result_dict = calculate_metrics(y_list, y_list_pred)

        # Print verbose information if required
        if need_verbose:
            log_results(name, result_dict, epoch_start_time)

    return result_dict


def calculate_metrics(y_true, y_pred):
    """Calculates evaluation metrics: MSE, R, R^2, RMSE, and MAPE."""
    mean_pred = y_pred.mean(axis=0)
    std_pred = y_pred.std(axis=0)
    mean_true = y_true.mean(axis=0)
    std_true = y_true.std(axis=0)

    # Filter non-zero std values to avoid division errors
    valid_indices = (std_true != 0)
    corr = np.mean(((y_pred - mean_pred) * (y_true - mean_true))[:,valid_indices] / (
            std_pred[valid_indices] * std_true[valid_indices]))

    mse = np.mean(np.square(y_pred - y_true))
    rmse = np.sqrt(mse)
    r_squared = corr ** 2

    # Calculate MAPE
    not_zero_idx = y_true != 0.0
    mape = np.mean(np.abs((y_pred[not_zero_idx] - y_true[not_zero_idx]) / np.abs(y_true[not_zero_idx]))) * 100

    return {'mse': mse, 'r': corr, 'r^2': r_squared, 'rmse': rmse, 'mape': mape}


def log_results(name, result_dict, start_time):
    """Logs the results and the time taken for the epoch."""
    epoch_time = time.time() - start_time
    print(
        f"{name} corr = {result_dict['r']:.4f}, rmse = {result_dict['rmse']:.4f}, mape = {result_dict['mape']:.4f} %, time = {epoch_time:.4f} s")


def train(args, model, data_packet, probs=None, ts_data=None, device='cuda'):
    model.train(True)
    optimizer = Adam(model.parameters(), args.lr)
    loss_fun = nn.MSELoss(reduction='mean').to(device)

    best_mse = float('inf')
    best_r2 = 0.0
    best_mse_model, best_r2_model = None, None

    x_train, y_train = data_packet['x_train'], data_packet['y_train']
    x_valid, y_valid = data_packet['x_valid'], data_packet['y_valid']

    n_iterations = len(x_train) // args.batch_size
    samples_idx = np.arange(len(x_train))
    step_print_num = 30

    for epoch in range(args.num_epochs):
        epoch_start_time = time.time()
        model.train()
        shuffle_idx = np.random.permutation(samples_idx)
        idx_to_del = np.zeros(0, dtype=int)

        for iter in range(n_iterations):
            idx_ele, shuffle_idx, idx_to_del = get_sample_indices(args, shuffle_idx, idx_to_del, iter)
            idx_batch = get_batch_indices(args, probs, idx_ele, idx_to_del, shuffle_idx, samples_idx)

            X, Y = x_train[idx_batch], y_train[idx_batch]
            X, Y = to_device(X, Y, device)

            foma_X, foma_Y = get_foma_batch(args, X, Y)

            if args.foma_latent == 0:
                pred_Y = model.forward(foma_X)
            else:
                pred_Y, foma_Y = model.forward_foma(args, foma_X, foma_Y)

            loss = calculate_loss(args, loss_fun, pred_Y, foma_Y, ts_data)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            idx_to_del = np.append(idx_to_del, idx_batch).astype(int)

            if args.dataset == 'Dti_dg' and (iter - 1) % (n_iterations // step_print_num) == 0:
                result_dict, best_mse_model, best_r2_model, best_mse, best_r2 = validate_and_save(args, model, x_valid,
                                                                                                  y_valid, epoch,
                                                                                                  iter, n_iterations,
                                                                                                  epoch_start_time,
                                                                                                  best_mse, best_r2,
                                                                                                  best_mse_model,
                                                                                                  best_r2_model, device)

        result_dict, best_mse_model, best_r2_model, best_mse, best_r2 = validate_and_save(args, model, x_valid, y_valid,
                                                                                          epoch, None, None,
                                                                                          epoch_start_time, best_mse,
                                                                                          best_r2,
                                                                                          best_mse_model, best_r2_model,
                                                                                          device)

    return best_mse_model, best_r2_model, result_dict


def get_sample_indices(args, shuffle_idx, idx_to_del, iter):
    if args.batch_selection == 'random':
        idx_ele = iter
    else:
        idx_ele = shuffle_idx[0]
        shuffle_idx = shuffle_idx[~np.isin(shuffle_idx, idx_to_del)]
    return idx_ele, shuffle_idx, idx_to_del


def get_batch_indices(args, probs, idx_ele, idx_to_del, shuffle_idx, samples_idx):
    if args.batch_selection in ['knn', 'knnp']:
        probs_i = probs[idx_ele]

        if args.batch_selection == 'knnp':
            temp_ent = probs_i[idx_to_del]
            probs_i[idx_to_del] = 0
            sum_probs = np.sum(probs_i)

            idx_neigh = np.random.choice(samples_idx, size=args.batch_size - 1,
                                         replace=False, p=probs_i / sum_probs)

            probs_i[idx_to_del] = temp_ent

        elif args.batch_selection == 'knn':
            probs_i = probs_i[~np.isin(probs_i, idx_to_del)]
            idx_neigh = probs_i[1:args.batch_size]

        idx_neigh = np.insert(idx_neigh, 0, idx_ele).astype(int)

    else:  # random mix
        idx_neigh = shuffle_idx[idx_ele * args.batch_size:(idx_ele + 1) * args.batch_size]


    return idx_neigh


def to_device(X, Y, device):
    if isinstance(X, np.ndarray):
        X = torch.tensor(X, dtype=torch.float32).to(device)
        Y = torch.tensor(Y, dtype=torch.float32).to(device)
    elif not X.is_cuda:
        X = X.to(device)
        Y = Y.to(device)
    return X, Y


def get_foma_batch(args, X, Y):
    if args.foma_input:
        return get_batch_foma(args, X, Y)
    return X, Y


def calculate_loss(args, loss_fun, pred_Y, foma_Y, ts_data):
    if args.dataset == 'TimeSires':
        scale = ts_data.scale.expand(pred_Y.size(0), ts_data.m)
        return loss_fun(pred_Y * scale, foma_Y * scale)
    return loss_fun(pred_Y, foma_Y)


def validate_and_save(args, model, x_valid, y_valid, epoch, iter, n_iterations, epoch_start_time, best_mse, best_r2,
                      best_mse_model, best_r2_model, device):
    step_info = f', step = {(epoch * n_iterations + iter)}' if iter is not None else ''
    result_dict = test(args, model, x_valid, y_valid, f'Train epoch {epoch}{step_info}:\t', args.show_process,
                       epoch_start_time, device)
    if result_dict['mse'] <= best_mse:
        best_mse = result_dict['mse']
        best_mse_model = copy.deepcopy(model)
        print(f'update best mse! epoch = {epoch}')
    if result_dict['r'] ** 2 >= best_r2:
        best_r2 = result_dict['r'] ** 2
        best_r2_model = copy.deepcopy(model)
    return result_dict, best_mse_model, best_r2_model, best_mse, best_r2
