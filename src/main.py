import argparse
import os
import pickle
import time
import torch
import neptune

from data_generate import load_data
from utils import set_seed, get_unique_file_name, write_result, write_model, get_id, get_probabilities, get_config
import algorithm
from models import Learner, Learner_TimeSeries, Learner_Dti_dg, Learner_RCF_MNIST


def parse_arguments():
    """Parses and returns command-line arguments."""
    parser = argparse.ArgumentParser(description='foma')

    # Paths and dataset
    parser.add_argument('--result_root_path', type=str, default="../result/", help="Path to store the results")
    parser.add_argument('--dataset', type=str, default='NO2', help='Dataset name')

    # Training settings
    parser.add_argument('--seed', type=int, default=0, help="Random seed")
    parser.add_argument('--gpu', type=int, default=0, help="GPU device to use")
    parser.add_argument('--num_epochs', type=int, default=100, help="Number of epochs")
    parser.add_argument('--batch_size', type=int, default=128, help="Batch size")
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate")

    # FOMA settings
    parser.add_argument('--id_dir', type=str, default='./ids', help="Path for intrinsic dimension")
    parser.add_argument('--probabilities_dir', type=str, default='./probabilities', help="Path for probabilities")
    parser.add_argument('--batch_selection', type=str, default='knn', help="Batch selection method (knn, knnp, random)")
    parser.add_argument('--foma_input', type=int, default=1, help="Apply FOMA on input data")
    parser.add_argument('--foma_latent', type=int, default=0, help="Apply FOMA on latent data")
    parser.add_argument('--estimate_id', type=int, default=0,
                        help="Estimate intrinsic dimension (0: no, 1: entire dataset, 2: each batch)")
    parser.add_argument('--alpha', type=float, default=1, help="Alpha value for FOMA")
    parser.add_argument('--rho', type=float, default=.9, help="Rho value for FOMA")
    parser.add_argument('--small_singular', type=int, default=0, help="Scale the smallest or largest singular values")

    # Verbose and model settings
    parser.add_argument('--show_process', type=int, default=1, help="Show RMSE and R^2 during the process")
    parser.add_argument('--show_setting', type=int, default=1, help="Show settings")
    parser.add_argument('--read_best_model', type=int, default=0, help="Read the best model (0: no, 1: yes)")
    parser.add_argument('--store_model', type=int, default=1, help="Store the trained model or not")
    return parser.parse_args()


def setup_environment(args):
    """Sets up directories and initializes device based on arguments."""
    os.makedirs(args.id_dir, exist_ok=True)
    os.makedirs(args.probabilities_dir, exist_ok=True)
    os.makedirs(args.result_root_path, exist_ok=True)

    result_path = os.path.join(args.result_root_path, f"{args.dataset}/")
    os.makedirs(result_path, exist_ok=True)

    set_seed(args.seed)

    # Setup device
    if torch.cuda.is_available() and args.gpu != -1:
        torch.cuda.set_device(f'cuda:{args.gpu}')
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cpu')

    if args.show_setting:
        print(f"Device: {device}")

    return device, result_path


def load_model(args, ts_data, device):
    """Loads the appropriate model based on the dataset type."""
    if args.dataset == 'TimeSeries':
        model = Learner_TimeSeries(args=args, data=ts_data).to(device)
    elif args.dataset == 'Dti_dg':
        model = Learner_Dti_dg(hparams=None).to(device)
    elif args.dataset == 'RCF_MNIST':
        model = Learner_RCF_MNIST(args=args).to(device)
    else:
        model = Learner(args=args).to(device)

    if args.show_setting:
        nParams = sum([p.nelement() for p in model.parameters()])
        print(f'Number of parameters: {nParams}')

    return model


def train_model(args, data_packet, ts_data, device, result_path):
    """Trains the model and evaluates its performance."""
    all_begin = time.time()

    model = load_model(args, ts_data, device)
    X, Y = data_packet['x_train'], data_packet['y_train']

    # Get knn or sampling probabilities
    if args.estimate_id == 1:
        id_name = f'{args.dataset_name}_id.npy'
        id_path = os.path.join(args.id_dir, id_name)
        args.id = get_id(id_path, X, Y)
        print(f'Estimated intrinsic dimension = {args.id}')

    probs = get_probabilities(args, args.probabilities_dir, Y)

    # Train the model
    best_model_dict = {}
    best_model_dict['rmse'], best_model_dict['r'], result_dict = algorithm.train(
        args, model, data_packet, probs, ts_data, device
    )

    # Final evaluation
    result_dict_best = evaluate_model(args, best_model_dict, data_packet, device, all_begin)

    # Calculate worst accuracy and save results
    algorithm.cal_worst_acc(args, data_packet, best_model_dict[args.metrics], result_dict_best, all_begin, device)
    write_result(args, result_dict_best, result_path)

    # Log to Neptune and save the model
    args.run[f"test_{args.metrics}"] = result_dict_best[args.metrics]
    args.run[f"best_val_{args.metrics}"] = result_dict[args.metrics]
    if args.metrics == 'rmse':
        args.run["test_mape"] = result_dict_best['mape']
    if args.store_model:
        write_model(args, best_model_dict[args.metrics], result_path)

    return result_dict_best[args.metrics]


def evaluate_model(args, best_model_dict, data_packet, device, start_time):
    """Performs final evaluation of the best model."""
    print('=' * 30 + ' Single experiment result ' + '=' * 30)
    return algorithm.test(
        args,
        best_model_dict[args.metrics],
        data_packet['x_test'],
        data_packet['y_test'],
        f'seed = {args.seed}: Final test for best {args.metrics}, FOMA on input: {args.foma_input}, latent = {args.foma_latent}:\n',
        args.show_process,
        start_time,
        device
    )


def main():
    """Main function that orchestrates the training and evaluation process."""
    args = parse_arguments()
    args.cuda = torch.cuda.is_available()


    args = argparse.Namespace(**get_config(args))
    args.dataset_name = args.dataset


    device, result_path = setup_environment(args)

    data_packet, ts_data = load_data(args)

    if args.read_best_model == 0:
        train_model(args, data_packet, ts_data, device, result_path)
    else:
        load_and_verify_model(args, data_packet, result_path, device)



def load_and_verify_model(args, data_packet, result_path, device):
    """Loads a pre-trained model and verifies its performance."""
    pt_full_path = os.path.join(result_path, get_unique_file_name(args, '', '.pickle'))
    with open(pt_full_path, 'rb') as f:
        read_model = pickle.load(f)
    print(f'Loaded best model from {pt_full_path}')

    all_begin = time.time()
    print('=' * 30 + ' Read best model and verify result ' + '=' * 30)

    read_result_dic = algorithm.test(
        args, read_model, data_packet['x_test'], data_packet['y_test'],
        f'seed = {args.seed}: Final test for read model: {pt_full_path}:\n',
        True, all_begin, device
    )
    algorithm.cal_worst_acc(args, data_packet, read_model, read_result_dic, all_begin, device)
    write_result(args, 'read', read_result_dic, result_path)


if __name__ == '__main__':
    main()
