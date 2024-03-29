"""
This script serves as the manager for the project. It takes command-line arguments and performs the specified task.

Usage:
    python manager.py <task> [--time_limit <time_limit>] [--checkpoint_load <checkpoint_load>] [--checkpoint_save <checkpoint_save>] [--checkpoint_save_freq <checkpoint_save_freq>] [--dataset_path <dataset_path>]

Arguments:
    task (str): The task to perform. Currently, only "train" is supported.

Optional Arguments:
    time_limit (int): The time limit for training in minutes. The training will be saved at the end of the time limit. Default is 100 minutes.
    checkpoint_load (str): The path to a checkpoint file to load. Default is None.
    checkpoint_save (str): The path to save the checkpoint file. Default is None.
    checkpoint_save_freq (int): The frequency to save the checkpoint in epochs. Default is 10 epochs.
    dataset_path (str): The path to the dataset. Default is "data/".

Example:
    python manager.py train --time_limit 120 --checkpoint_save models/checkpoint.pth --dataset_path data/train/
"""

import argparse
import pathlib

import training_loop


def create_args() -> argparse.Namespace:
    """
    Create and parse command-line arguments.

    Returns:
        argparse.Namespace: The parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Manager")
    parser.add_argument("task", type=str, help="Task to perform")
    parser.add_argument(
        "--time_limit",
        type=int,
        default=100,
        help="Time limit for training in minutes will save at end of limit",
    )
    parser.add_argument(
        "--checkpoint_load", type=str, default=None, help="Checkpoint to load"
    )
    parser.add_argument(
        "--checkpoint_save", type=str, default=None, help="Checkpoint to save"
    )
    parser.add_argument(
        "--checkpoint_save_freq",
        type=int,
        default=10,
        help="Frequency to save the checkpoint in epochs.",
    )
    parser.add_argument(
        "--dataset_path", type=str, default="data/", help="Path to dataset"
    )

    return parser.parse_args()


def main():
    """
    Main function that executes the specified task based on the command-line arguments.
    """
    args = create_args()
    if args.task == "train":
        training_loop.train(args)
    else:
        raise ValueError("Task not recognized")


if __name__ == "__main__":
    main()
