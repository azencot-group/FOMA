import argparse
import datetime
import random
import json
import os
import sys
import csv
import copy
from collections import defaultdict

from tempfile import mkdtemp

import numpy as np
import torch
import torch.optim as optim

import models
from model import Model as poverty_model
from models.poverty import getTrainDataLoader, getDataLoaders
from config import dataset_defaults
from utils import Logger, save_best_model, return_predict_fn, get_data_unshuffeled, get_id, \
    get_probabilities, save_pred
from foma import get_batch_foma

# code base: https://github.com/huaxiuyao/LISA/tree/main/domain_shifts

runId = datetime.datetime.now().isoformat().replace(':', '_')
torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(description='Gradient Matching for Domain Generalization.')
# General
parser.add_argument('--dataset', type=str, default='poverty',
                    help="Name of dataset")
parser.add_argument('--algorithm', type=str, default='erm',
                    help='training scheme, choose between fish or erm.')
parser.add_argument('--experiment', type=str, default='.',
                    help='experiment name, set as . for automatic naming.')
parser.add_argument('--experiment_dir', type=str, default='../',
                    help='experiment directory')
parser.add_argument('--data-dir', type=str, default='./',
                    help='path to data dir')
# Computation
parser.add_argument('--nocuda', type=int, default=False,
                    help='disables CUDA use')
parser.add_argument('--seed', type=int, default=0,
                    help='random seed, set as -1 for random.')
parser.add_argument("--print_loss_iters", default=100, type=int)

parser.add_argument("--save_pred", default=False, action='store_true')
parser.add_argument("--save_dir", default='result', type=str)
parser.add_argument("--fold", default='A', type=str)
parser.add_argument('--epochs', type=int, default=2,
                    help='number of epochs to train')
parser.add_argument('--lr', type=float, default=1e-3,
                    help="learning rate")
parser.add_argument('--weight_decay', type=float, default=0,
                    help="weight decay")
parser.add_argument('--batch_size', type=int, default=64,
                    help='batch size')

#### FOMA ####
parser.add_argument('--id_dir', type=str, default='./ids',
                    help="path to store the intrinsic dimension of the dataset")
parser.add_argument('--probabilities_dir', type=str, default='./probabilities',
                    help="path to store the knn or knn based probabilities of the dataset")

parser.add_argument('--batch_selection', type=str, default='knn',
                    help="knn, knnp or random")
parser.add_argument('--foma_input', type=int, default=1,
                    help='apply foma on input data')
parser.add_argument('--foma_latent', type=int, default=0,
                    help='apply foma on latent data')
parser.add_argument('--estimate_id', type=int, default=1,
                    help='estimate intrinsic dimension: 0 for no use, 1 for id estimated on entire dataset, 2 for id estimated on each batch')
parser.add_argument('--alpha', type=float, default=0.8,
                    help="alpha value for foma")
parser.add_argument('--rho', type=float, default=.9,
                    help="rho value for foma")
parser.add_argument('--small_singular', type=int, default=0,
                    help="scale the smallest singular values or the largest singular values")

args = parser.parse_args()

args.cuda = not args.nocuda and torch.cuda.is_available()
device = torch.device("cuda")
if args.nocuda:
    print(f'use cpu')
    device = torch.device("cpu")

dict_name = args.dataset
args_dict = args.__dict__

dataset_config = dataset_defaults.get(dict_name, {})
args_dict.update(dataset_config)
args_dict.update(dataset_config['command_args'])
args = argparse.Namespace(**args_dict)

# random select a training fold according to seed. Can comment this line and set args.fold manually as well
args.fold = ['A', 'B', 'C', 'D', 'E'][args.seed % 5]

if args.save_pred:
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

##### set seed #####
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.manual_seed(args.seed)
print(f'args.seed = {args.seed}')
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# experiment directory setup
args.experiment = f"{args.dataset}_{args.algorithm}_{args.seed}" \
    if args.experiment == '.' else args.experiment
directory_name = '{}/experiments/{}'.format(args.experiment_dir, args.experiment)

if not os.path.exists(directory_name):
    os.makedirs(directory_name)
runPath = mkdtemp(prefix=runId, dir=directory_name)

# logging setup
sys.stdout = Logger('{}/run.log'.format(runPath))
print('RunID:' + runPath)
print(args)
with open('{}/args.json'.format(runPath), 'w') as fp:
    json.dump(args.__dict__, fp)
torch.save(args, '{}/args.rar'.format(runPath))

# load model
model = poverty_model(args, weights=None).to(device)

train_loader, tv_loaders = getDataLoaders(args, device=device)
val_loader, test_loader = tv_loaders['val'], tv_loaders['test']

print(
    f'len(train_loader) = {len(train_loader)}, len(val_loader) = {len(val_loader)}, len(test_loader) = {len(test_loader)}')

n_class = getattr(models, f"{args.dataset}_n_class")

assert args.optimiser in ['SGD', 'Adam', 'AdamW'], "Invalid choice of optimiser, choose between 'Adam' and 'SGD'"
opt = getattr(optim, args.optimiser)

params = filter(lambda p: p.requires_grad, model.parameters())
optimiserC = opt(params, lr=args.lr, weight_decay=args.weight_decay)

predict_fn = return_predict_fn(args.dataset)


def train(x_train, y_train, epoch, knn, agg):
    model.train()
    print('\n====> Epoch: {:03d} '.format(epoch))

    criterion = torch.nn.MSELoss(reduction='mean').to(device)
    idx_to_del = np.zeros(0)
    shuffle_idx = np.random.permutation(np.arange(len(x_train)))
    iteration = int(np.ceil(len(x_train) / args.batch_size))
    samples_idx = np.arange(x_train.shape[0])

    if args.batch_selection == 'knnp' or args.batch_selection == 'knn':
        knn_i = copy.deepcopy(knn)

    for i in range(iteration):
        idx_ele = shuffle_idx[0]
        if args.batch_selection == 'knnp' or args.batch_selection == 'knn':
            knn_idx = knn_i[idx_ele]

        if args.batch_selection == 'knn':
            if len(idx_to_del) > 0:
                knn_idx = knn_idx[
                    ~np.isin(knn_idx, idx_to_del)]
            idx_neigh = knn_idx[1:args.batch_size]
        else:
            sample_size = min(args.batch_size - 1, len(shuffle_idx))
            if args.batch_selection == 'knnp':
                if len(idx_to_del) > 0:
                    knn_idx[idx_to_del] = 0
                    knn_idx /= np.sum(knn_idx)

                idx_neigh = np.random.choice(samples_idx, size=sample_size, replace=False,
                                             p=knn_idx)

            if args.batch_selection == 'random':
                idx_neigh = np.random.choice(shuffle_idx, size=sample_size, replace=False)

        idx_batch = np.insert(idx_neigh, 0, idx_ele).astype(int)

        X = x_train[idx_batch]
        Y = y_train[idx_batch]

        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float32).to(device)
            Y = torch.tensor(Y, dtype=torch.float32).to(device)

        elif not x_train.is_cuda:
            X = X.to(device)
            Y = Y.to(device)

        mixup_X = X
        mixup_Y = Y

        # forward
        if args.foma_input:
            mixup_X, mixup_Y = get_batch_foma(args, X, Y)

        if args.foma_latent == 0:
            pred_Y = model.forward(mixup_X)
        else:
            pred_Y, mixup_Y = model.forward_foma(args, mixup_X, mixup_Y)

        loss = criterion(pred_Y, mixup_Y)

        # backward
        optimiserC.zero_grad()
        loss.backward()
        optimiserC.step()

        if (i + 1) % args.print_iters == 0 and args.print_iters != -1 and (
                (i + 1) / args.print_iters >= 2 or epoch >= 1):
            print(f'iteration {(i + 1):05d}: ')
            test(val_loader, agg, loader_type='val')
            test(test_loader, agg, loader_type='test')
            save_best_model(model, runPath, agg, args)
            model.train()


def test(test_loader, agg, loader_type='test', verbose=True, save_ypred=False, save_dir=None):
    model.eval()
    yhats, ys, metas = [], [], []
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            # get the inputs
            x, y = batch[0].to(device), batch[1].to(device)
            y_hat = model(x)
            ys.append(y)
            yhats.append(y_hat)
            metas.append(batch[2])

        ypreds, ys, metas = predict_fn(torch.cat(yhats)), torch.cat(ys), torch.cat(metas)
        if save_ypred:  # random select a fold
            save_name = f"{args.dataset}_split:{loader_type}_fold:" \
                        f"{['A', 'B', 'C', 'D', 'E'][args.seed % 5]}" \
                        f"_epoch:best_pred.csv"
            with open(f"{runPath}/{save_name}", 'w') as f:
                writer = csv.writer(f)
                writer.writerows(ypreds.unsqueeze(1).cpu().tolist())
        test_val = test_loader.dataset.eval(ypreds.cpu(), ys.cpu(), metas)

        agg[f'{loader_type}_stat'].append(test_val[0][args.selection_metric])
        if verbose:
            print(f"=============== {loader_type} ===============\n{test_val[-1]}")


if __name__ == '__main__':
    # set learning rate schedule
    if args.scheduler == 'StepLR':
        scheduler = optim.lr_scheduler.StepLR(optimiserC, step_size=1, **args.scheduler_kwargs)
        scheduler.step_every_batch = False
        scheduler.use_metric = False
    else:
        scheduler = None

    print("=" * 30 + f" Training: {args.algorithm} for {args.dataset} " + "=" * 30)
    unshuffled_train_loader = getTrainDataLoader(args, device=device)
    X, Y = get_data_unshuffeled(unshuffled_train_loader)

    if not os.path.exists(args.id_dir):
        os.mkdir(args.id_dir)
    if not os.path.exists(args.probabilities_dir):
        os.mkdir(args.probabilities_dir)

    id_name = f'poverty_id_{args.fold}.npy'
    id_path = os.path.join(args.id_dir, id_name)
    if args.estimate_id == 1:
        args.id = get_id(id_path, X, Y)
        print(f'Estimated intrinsic dimension = {args.id}')

    #### data knn ####
    probs = get_probabilities(args, args.probabilities_dir, Y)

    agg = defaultdict(list)
    agg['val_stat'] = [0.]
    agg['test_stat'] = [0.]

    for epoch in range(args.epochs):
        train(X, Y, epoch, probs, agg)
        test(val_loader, agg, 'val', True)
        if scheduler is not None:
            scheduler.step()

        test(test_loader, agg, 'test', True)
        save_best_model(model, runPath, agg, args)

        if args.save_pred:
            save_pred(args, model, train_loader, epoch, args.save_dir, predict_fn, device)

    model.load_state_dict(torch.load(runPath + '/model.rar'))
    print('Finished training! Loading best model...')
    for split, loader in tv_loaders.items():
        test(loader, agg, loader_type=split, verbose=True, save_ypred=False)
