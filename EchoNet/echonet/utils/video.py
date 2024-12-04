"""Functions for training and running EF prediction."""

import math
import os
import time
import copy
import click
import numpy as np
import sklearn.metrics
import torch
import torchvision
import tqdm
import random

import echonet
from argparse import Namespace
from utils import get_id, get_probabilities
from models import VideoNet
from foma import get_batch_foma
import neptune
from neptune.utils import stringify_unsupported

@click.command("video")
@click.option("--data_dir", type=click.Path(exists=True, file_okay=False), default=None)
@click.option("--output", type=click.Path(file_okay=False), default=None)
@click.option("--result_root_path", type=click.Path(file_okay=False), default="../result/")
@click.option("--task", type=str, default="EF")
@click.option("--dataset_name", type=str, default="echo")
@click.option("--model_name", type=click.Choice(
    sorted(name for name in torchvision.models.video.__dict__
           if name.islower() and not name.startswith("__") and callable(torchvision.models.video.__dict__[name]))),
              default="r2plus1d_18")
@click.option("--pretrained/--random", default=True)
@click.option("--weights", type=click.Path(exists=True, dir_okay=False), default=None)
@click.option("--run_test", default=True)
@click.option("--num_epochs", type=int, default=1)
@click.option("--lr", type=float, default=1e-4)
@click.option("--weight_decay", type=float, default=1e-4)
@click.option("--lr_step_period", type=int, default=15)
@click.option("--frames", type=int, default=32)
@click.option("--period", type=int, default=2)
@click.option("--num_train_patients", type=int, default=None)
@click.option("--num_workers", type=int, default=0)  # 4)
@click.option("--batch_size", type=int, default=10)
@click.option("--device", type=str, default=None)
@click.option("--seed", type=int, default=0)
@click.option("--train", type=int, default=1)
@click.option("--id_dir", type=click.Path(exists=True, file_okay=False), default="./ids", help="Path for intrinsic dimension")
@click.option("--probabilities_dir", type=click.Path(exists=True, file_okay=False), default="./probabilities", help="Path for probabilities")
@click.option("--batch_selection", type=str, default="knn", help="Batch selection method (knn, knnp, random)")
@click.option("--foma_input", type=int, default=0, help="Apply FOMA on input data")
@click.option("--foma_latent", type=int, default=1, help="Apply FOMA on latent data")
@click.option("--estimate_id", type=click.Choice(["0", "1", "2"], case_sensitive=False), default="0", help="Estimate intrinsic dimension (0: dont estimate, 1: entire dataset, 2: each batch)")
@click.option("--alpha", type=float, default=1.1, help="Alpha value for FOMA")
@click.option("--rho", type=float, default=0.85, help="Rho value for FOMA")
@click.option("--small_singular", type=int, default=1, help="Scale the smallest or largest singular values")

def run(
        data_dir=None,
        output=None,
        result_root_path="../result/",
        task="EF",
        dataset_name="echo",
        model_name="r2plus1d_18",
        pretrained=True,
        weights=None,
        run_test=True,
        num_epochs=1,
        lr=1e-4,
        weight_decay=1e-4,
        lr_step_period=15,
        frames=32,
        period=2,
        num_train_patients=None,
        num_workers=0,
        batch_size=10,
        device=None,
        seed=0,
        train=1,
        id_dir="./ids",
        probabilities_dir="./probabilities",
        batch_selection="knn",
        foma_input=0,
        foma_latent=1,
        estimate_id="0",
        alpha=1.1,
        rho=0.85,
        small_singular=1,
):
    data_dir = "../../EchoNet-Dynamic"
    """Trains/tests EF prediction model.

    \b
    Args:
        data_dir (str, optional): Directory containing dataset. Defaults to
            `echonet.config.DATA_DIR`.
        output (str, optional): Directory to place outputs. Defaults to
            output/video/<model_name>_<pretrained/random>/.
        task (str, optional): Name of task to predict. Options are the headers
            of FileList.csv. Defaults to ``EF''.
        model_name (str, optional): Name of model. One of ``mc3_18'',
            ``r2plus1d_18'', or ``r3d_18''
            (options are torchvision.models.video.<model_name>)
            Defaults to ``r2plus1d_18''.
        pretrained (bool, optional): Whether to use pretrained weights for model
            Defaults to True.
        weights (str, optional): Path to checkpoint containing weights to
            initialize model. Defaults to None.
        run_test (bool, optional): Whether or not to run on test.
            Defaults to False.
        num_epochs (int, optional): Number of epochs during training.
            Defaults to 45.
        lr (float, optional): Learning rate for SGD
            Defaults to 1e-4.
        weight_decay (float, optional): Weight decay for SGD
            Defaults to 1e-4.
        lr_step_period (int or None, optional): Period of learning rate decay
            (learning rate is decayed by a multiplicative factor of 0.1)
            Defaults to 15.
        frames (int, optional): Number of frames to use in clip
            Defaults to 32.
        period (int, optional): Sampling period for frames
            Defaults to 2.
        n_train_patients (int or None, optional): Number of training patients
            for ablations. Defaults to all patients.
        num_workers (int, optional): Number of subprocesses to use for data
            loading. If 0, the data will be loaded in the main process.
            Defaults to 4.
        device (str or None, optional): Name of device to run on. Options from
            https://pytorch.org/docs/stable/tensor_attributes.html#torch.torch.device
            Defaults to ``cuda'' if available, and ``cpu'' otherwise.
        batch_size (int, optional): Number of samples to load per batch
            Defaults to 20.
        seed (int, optional): Seed for random number generator. Defaults to 0.
    """

    # Seed RNGs
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    args = Namespace(**locals())

    nept_run = neptune.init_run(project="azencot-group/Soma-ablation")
    param_dict = vars(args)
    nept_run["params"] = stringify_unsupported(param_dict)
    # Set default output directory
    if output is None:
        output = os.path.join("../../output", "video", "Seed{}_{}_{}_{}_{}".format(seed
                                                                                   , model_name,
                                                                                   frames, period,
                                                                                   "pretrained" if pretrained else "random"))
    os.makedirs(output, exist_ok=True)

    print(os.getcwd())

    id_dir = '/home/ilyakau/SOMA/EchoNet/echonet/utils/ids'
    probabilities_dir = '/cs/cs_groups/azencot_group/SOMA/probabilities'
    if not os.path.exists(id_dir):
        os.mkdir(id_dir)
    if not os.path.exists(probabilities_dir):
        os.mkdir(probabilities_dir)


    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif device == 'cpu':
        device = torch.device("cpu")
    elif device == 'cuda':
        device = torch.device("cuda")

    print(locals())
    print(f'device = {device}, batch_size = {batch_size}, num_epochs = {num_epochs}, num_workers = {num_workers}')
    print(f'input mixup = {foma_input}, manifold mixup = {foma_latent}, alpha = {alpha}')
    print(f'output = {output}')
    print(f'run_test = {run_test}')
    # Set up model

    model = torchvision.models.video.__dict__[args.model_name](pretrained=args.pretrained)

    model.fc = torch.nn.Linear(model.fc.in_features, 1)
    model.fc.bias.data[0] = 55.6
    model = VideoNet(model, args)

    # if device.type == "cuda":
    #     model = torch.nn.DataParallel(model)
    model.to(device)

    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    print('finish model definition')

    if weights is not None:
        checkpoint = torch.load(weights)
        model.load_state_dict(checkpoint['state_dict'])

    # Set up optimizer
    optim = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    if lr_step_period is None:
        lr_step_period = math.inf
    scheduler = torch.optim.lr_scheduler.StepLR(optim, lr_step_period)

    # Compute mean and std
    mean, std = echonet.utils.get_mean_and_std(echonet.datasets.Echo(root=data_dir, split="train"))
    kwargs = {"target_type": task,
              "mean": mean,
              "std": std,
              "length": frames,
              "period": period,
              }

    # Set up datasets and dataloaders
    dataset = {}
    dataset["train"] = echonet.datasets.Echo(root=data_dir, split="train", **kwargs, pad=12)
    if num_train_patients is not None and len(dataset["train"]) > num_train_patients:
        # Subsample patients (used for ablation experiment)
        indices = np.random.choice(len(dataset["train"]), num_train_patients, replace=False)
        dataset["train"] = torch.utils.data.Subset(dataset["train"], indices)

    dataset["val"] = echonet.datasets.Echo(root=data_dir, split="val", **kwargs)
    dataset["test"] = echonet.datasets.Echo(root=data_dir, split="test", **kwargs)

    id_name = f'{args.dataset_name}_id.npy'
    id_path = os.path.join(id_dir, id_name)

    f_name = f'{args.dataset_name}_{args.batch_selection}.npy'
    probs_path = os.path.join(probabilities_dir, f_name)
    probabilities = None

    Y = None

    if not os.path.exists(id_path) or not os.path.exists(probs_path):
        print(f'ids - {os.path.exists(id_path)}, probs - {os.path.exists(probs_path)}')
        dataloader = torch.utils.data.DataLoader(
            dataset["train"], batch_size=512, num_workers=8, shuffle=False, drop_last=False)
        X = []
        Y = []
        for (_, yi) in tqdm.tqdm(dataloader, desc="Processing Batches"):
            Y.append(yi)
        Y = torch.hstack(Y)


    print('Estimating id')
    if args.estimate_id == 1:
        args.id = get_id(id_path, X, Y)

    print('Estimating probabilities')
    probabilities = get_probabilities(args, probabilities_dir, Y)

    del Y

    len_list_dic = {}
    for phase in ["train", "val", "test"]:
        len_list_dic[phase] = len(dataset[phase])
        len_list_dic[phase] -= (len_list_dic[phase] % batch_size)  # extra part

    ### get y dataset [for length] ###
    flag = 0

    val_loader = torch.utils.data.DataLoader(
        dataset["val"], batch_size=64, num_workers=num_workers, shuffle=False, pin_memory=(device.type == "cuda"))
    # Run training and testing loops
    epoch_resume = 0
    bestLoss = float("inf")

    best_state_dict = model.state_dict()
    ds = dataset['train']
    for epoch in range(epoch_resume, num_epochs):
        print("Epoch #{}".format(epoch), flush=True)
        start_time = time.time()
        for i in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(i)


        train_loss = train_foma(args, model, len_list_dic['train'], ds, optim, device,
                                        probabilities)
        val_loss, _, _ = test(model, val_loader, device)
        nept_run["train/loss"].log(np.sqrt(train_loss))
        nept_run["val/loss"].log(np.sqrt(val_loss))

        scheduler.step()
        torch.cuda.empty_cache()

        # Save checkpoint
        # save = {
        #     'epoch': epoch,
        #     'state_dict': model.state_dict(),
        #     'period': period,
        #     'frames': frames,
        #     'best_loss': bestLoss,
        #     'loss': val_loss,
        #     'opt_dict': optim.state_dict(),
        #     'scheduler_dict': scheduler.state_dict(),
        # }
        # torch.save(save, os.path.join(output, "checkpoint.pt"))
        # print(f'save model to {output + "checkpoint.pt"}')
        if val_loss < bestLoss:
            # torch.save(save, os.path.join(output, "best.pt"))
            best_state_dict = copy.deepcopy(model.state_dict())
            bestLoss = val_loss

    # Load best weights
    if num_epochs != 0:
        # checkpoint = torch.load(os.path.join(output, "best.pt"))
        # model.load_state_dict(checkpoint['state_dict'])
        model.load_state_dict(best_state_dict)
        # print("Best validation loss {} from epoch {}\n".format(checkpoint["loss"], checkpoint["epoch"]))

    if run_test:
        for split in ["test"]:  # ["val", "test"]:
            # Performance without test-time augmentation

            # one clip
            """loss, yhat, y = ...
            f.write("{} (one clip) R2:   {:.3f} ({:.3f} - {:.3f})\n".format(split, *echonet.utils.bootstrap(y, yhat, sklearn.metrics.r2_score)))
            f.write("{} (one clip) MAE:  {:.2f} ({:.2f} - {:.2f})\n".format(split, *echonet.utils.bootstrap(y, yhat, sklearn.metrics.mean_absolute_error)))
            f.write("{} (one clip) MAPE: {:.5f} ({:.5f} - {:.5f})\n".format(split, *echonet.utils.bootstrap(y, yhat, sklearn.metrics.mean_absolute_percentage_error)))
            f.write("{} (one clip) RMSE: {:.2f} ({:.2f} - {:.2f})\n".format(split, *tuple(map(math.sqrt, echonet.utils.bootstrap(y, yhat, sklearn.metrics.mean_squared_error)))))
            f.flush()"""

            # Performance with test-time augmentation
            ds = echonet.datasets.Echo(root=data_dir, split=split, **kwargs, clips="all")
            dataloader = torch.utils.data.DataLoader(
                ds, batch_size=1, num_workers=num_workers, shuffle=False, pin_memory=(device.type == "cuda"))

            # all clips
            #
            test_loss, yhat, y = test(model, dataloader, device, save_all=True,
                                      block_size=batch_size)

            test_mape = echonet.utils.bootstrap(y,
                                                np.array(list(map(lambda x: x.mean(), yhat))),
                                                sklearn.metrics.mean_absolute_percentage_error)[0]

            test_rmse = rmse_value = tuple(map(math.sqrt, echonet.utils.bootstrap(
                y, np.array(list(map(lambda x: x.mean(), yhat))), sklearn.metrics.mean_squared_error)))[0]

            print("{} (all clips) MAPE: {:.5f} ({:.5f} - {:.5f})\n".format(split, *echonet.utils.bootstrap(y,
                                                                                                           np.array(
                                                                                                               list(
                                                                                                                   map(lambda
                                                                                                                           x: x.mean(),
                                                                                                                       yhat))),
                                                                                                           sklearn.metrics.mean_absolute_percentage_error)))
            print("{} (all clips) RMSE: {:.2f} ({:.2f} - {:.2f})\n".format(split, *tuple(map(math.sqrt,
                                                                                             echonet.utils.bootstrap(
                                                                                                 y, np.array(list(
                                                                                                     map(lambda
                                                                                                             x: x.mean(),
                                                                                                         yhat))),
                                                                                                 sklearn.metrics.mean_squared_error)))))

            nept_run["test/loss"].log(test_rmse)
            nept_run["test/mape"].log(test_mape)


def test(model, dataloader, device, save_all=False, block_size=None):
    """
    Run one epoch of evaluation for segmentation.

    Args:
        model (torch.nn.Module): Model to evaluate.
        dataloader (torch.utils.data.DataLoader): Dataloader for dataset.
        device (torch.device): Device to run on.
        save_all (bool, optional): If True, return predictions for all test-time augmentations separately.
                                   If False, return only the mean prediction. Defaults to False.
        block_size (int, optional): Maximum number of augmentations to run at the same time to limit memory usage.
                                    If None, run all augmentations simultaneously. Default is None.

    Returns:
        float: Average mean squared error loss.
        np.ndarray: Predicted outcomes.
        np.ndarray: Ground truth outcomes.
    """

    model.eval()  # Set the model to evaluation mode
    total_loss = 0  # Accumulate total loss
    total_samples = 0  # Count total number of samples
    sum_outcomes = 0  # Sum of ground truth outcomes (for variance computation)
    sum_squared_outcomes = 0  # Sum of squared ground truth outcomes

    yhat, y = [], []  # Lists to store predictions and ground truth

    with torch.no_grad():
        with tqdm.tqdm(total=len(dataloader)) as pbar:
            for X, outcome in dataloader:
                # Move data to the specified device
                X, outcome = X.to(device), outcome.to(device)

                # Store ground truth outcomes
                y.append(outcome.cpu().numpy())

                # Handle multiple clips by flattening if necessary
                if len(X.shape) == 6:  # Check for multi-clip input
                    batch_size, n_clips, c, f, h, w = X.shape
                    X = X.view(-1, c, f, h, w)  # Reshape for processing

                # Update statistics for variance calculation
                sum_outcomes += outcome.sum().item()
                sum_squared_outcomes += (outcome ** 2).sum().item()

                # Run model inference, handling block_size if specified
                if block_size is None:
                    outputs = model(X)
                else:
                    outputs = torch.cat([
                        model(X[i:i + block_size]) for i in range(0, X.size(0), block_size)
                    ])


                # Store predictions
                if save_all:
                    yhat.append(outputs.view(-1).cpu().numpy())
                else:
                    if len(X.shape) == 6:  # Reshape back to batch size if averaged
                        outputs = outputs.view(batch_size, n_clips, -1).mean(1)

                    yhat.append(outputs.view(-1).cpu().numpy())

                # Compute loss for the batch
                loss = torch.nn.functional.mse_loss(outputs.view(-1), outcome)
                total_loss += loss.item() * X.size(0)  # Accumulate weighted loss
                total_samples += X.size(0)  # Increment sample count

                # Update progress bar with current loss and variance
                mean_outcome = sum_outcomes / total_samples
                variance_outcome = sum_squared_outcomes / total_samples - mean_outcome ** 2
                pbar.set_postfix_str(f"Loss: {total_loss / total_samples:.2f}, Variance: {variance_outcome:.2f}")
                pbar.update()

    # Concatenate predictions and ground truth
    yhat = np.concatenate(yhat) if not save_all else yhat
    y = np.concatenate(y)

    return total_loss / total_samples, yhat, y


def train_foma(args, model, dataset_all_num, dataset, optim, device, probs=None):
    model.train()

    total = 0  # total training loss
    n = 0  # number of videos processed

    ### personal interation ###
    iteration = dataset_all_num // args.batch_size
    samples_idx = np.arange(dataset_all_num)
    shuffle_idx = np.random.permutation(samples_idx)
    idx_to_del = []
    for iter in range(iteration):
        idx_ele, shuffle_idx, idx_to_del = get_sample_indices(args, shuffle_idx, idx_to_del, iter)
        idx_batch = get_batch_indices(args, probs, idx_ele, idx_to_del, shuffle_idx, samples_idx)

        X = np.concatenate([dataset[i][0][np.newaxis, :] for i in idx_batch])
        outcome = np.concatenate([[dataset[i][1]] for i in idx_batch])

        X = torch.tensor(X)
        outcome = torch.tensor(outcome)

        X = X.to(device)
        outcome = outcome.to(device)

        average = (len(X.shape) == 6)

        if average:
            batch, n_clips, c, f, h, w = X.shape
            X = X.view(-1, c, f, h, w)

        if args.use_input_mixup:
            X, outcome = get_batch_foma(args, X, outcome)

        if args.use_manifold == 0:
            outputs = model.forward(X)
        else:
            outputs, outcome = model.forward_mixup(X, outcome)

        if average:
            outputs = outputs.view(batch, n_clips, -1).mean(1)

        loss = torch.nn.functional.mse_loss(outputs.view(-1), outcome)
        idx_to_del = np.append(idx_to_del, idx_batch).astype(int)

        optim.zero_grad()
        loss.backward()
        optim.step()

        total += loss.item() * X.size(0)
        n += X.size(0)

    return total / n

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


def stats_values(targets, flag=False):
    mean = torch.mean(targets)
    min = torch.min(targets)
    max = torch.max(targets)
    std = torch.std(targets)
    if flag:
        print(f'y stats: mean = {mean}, max = {max}, min = {min}, std = {std}')
    return mean, min, max, std


if __name__ == "__main__":
    run()
