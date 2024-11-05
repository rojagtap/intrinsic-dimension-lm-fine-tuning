import argparse

import numpy as np

from .plotter import plot_intrinsic_dimensions


def count_params(model):
    total_params = 0
    for param in model.parameters():
        if param.requires_grad:
            total_params += param.numel()

    return total_params


def generate_intrinsic_dimension_candidates(start, end, n_points=100, scale='log', plot=False, basedir=''):
    if scale == 'log':
        d_values = np.logspace(start, end, num=n_points, dtype=int)
    elif scale == 'linear':
        d_values = np.linspace(start, end, num=n_points, dtype=int)
    else:
        raise AttributeError(f'Unsupported scale {scale}')

    if plot:
        plot_intrinsic_dimensions(d_values, basedir, scale=scale)

    return d_values


def parse_args(model_name, dataset_name):
    parser = argparse.ArgumentParser(description=f"Fine-tune {model_name} on the {dataset_name} dataset")

    # Hyperparameters
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate for training")
    parser.add_argument("--train_batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--eval_batch_size", type=int, default=16, help="Batch size for evaluation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--num_epochs", type=float, default=3.0, help="Number of training epochs")
    parser.add_argument("--lr_scheduler_type", type=str, default="linear", help="Scheduler type")

    return parser.parse_args()
