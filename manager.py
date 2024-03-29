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
import os
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
        "--checkpoint_dir", type=str, default='opt-finetuned-icd9', help="Checkpoint directory to save in"
    )
    parser.add_argument(
        "--train_dataset", type=str, default="data/train_9.csv", help="The training dataset"
    )

    parser.add_argument(
        "--val_dataset", type=str, default="data/val_9.csv", help="The evaluation dataset"
    )

    parser.add_argument(
        "--test_dataset", type=str, default="data/test_9.csv", help="The test dataset"
    )

    parser.add_argument(
        "--code_labels", type=str, default="data/icd9_codes.csv", help="The training dataset code labels"
    )

    parser.add_argument(
        "--save_interval", type=int, default=1000, help="Save interval in steps for checkpoints."
    )
    
    # Wandb api key (optional skip wandb usage if not provided)
    parser.add_argument(
        "--wandb_key", type=str, default=None help="Wandb API key"
    )

    parser.add_arugment(
        "--run_name", type=str, default="run", help="Name of the run"
    )

    parser.add_arugment(
        "--project_name", type=str, default="OPT-Finetuning ICD9", help="Name of the run for Wandb"
    )
    parser.add_argument(
        "--fresh_start", action="store_true", help="Start fresh without loading checkpoint"
    )

    return parser.parse_args()


def main():
    """
    Main function that executes the specified task based on the command-line arguments.
    """
    args = create_args()
    if args.task == "train":
        os.environ["WANDB_PROJECT"]=args.project_name
        os.environ["WANDB_API_KEY"]=args.wandb_key
        training_loop.train(args)
    else:
        raise ValueError("Task not recognized")


if __name__ == "__main__":
    main()
