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
        "--checkpoint_dir",
        type=str,
        default="opt-finetuned-icd9-350m",
        help="Checkpoint directory to save in",
    )
    parser.add_argument(
        "--train_path",
        type=str,
        default="data/train_9.csv",
        help="The path to the training dataset",
    )

    parser.add_argument(
        "--val_path",
        type=str,
        default="data/val_9.csv",
        help="The path to the evaluation dataset",
    )

    parser.add_argument(
        "--test_path",
        type=str,
        default="data/test_9.csv",
        help="The path to the test dataset",
    )

    parser.add_argument(
        "--code_labels",
        type=str,
        default="data/icd9_codes.csv",
        help="The training dataset code labels",
    )

    parser.add_argument(
        "--save_interval",
        type=int,
        default=1,
        help="Save interval in epochs for checkpoints.",
    )

    # Wandb api key (optional skip wandb usage if not provided)
    parser.add_argument("--wandb_key", type=str, default=None, help="Wandb API key")
    parser.add_argument(
        "--wandb", action="store_true", help="Use currently logged in wandb user"
    )

    parser.add_argument("--run_name", type=str, default="run", help="Name of the run")

    parser.add_argument(
        "--project_name",
        type=str,
        default="OPT-Finetuning ICD9 350m",
        help="Name of the run for Wandb",
    )
    parser.add_argument(
        "--fresh_start",
        action="store_true",
        help="Start fresh without loading checkpoint",
    )

    parser.add_argument(
        "--cache_dir",
        type=str,
        default="data/cache",
        help="Dataset cache directory.",
    )

    parser.add_argument(
        "--disable_tqdm", action="store_true", help="Disable tqdm in output"
    )

    parser.add_argument(
        "--epochs", type=int, default=30, help="Number of epochs to train the model for"
    )

    parser.add_argument(
        "--n_trials", type=int, default=15, help="Number of hyperparam search trials"
    )

    parser.add_argument(
        "--tiny", action="store_true", help="Use a tiny subset of dataset for training"
    )

    parser.add_argument(
        "--search",
        action="store_true",
        help="Run hyperparam search (requires fresh_start)",
    )

    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Use gradient checkpointing",
    )

    parser.add_argument(
        "--biotech",
        action="store_true",
        help="Use the events_classification_biotech dataset",
    )

    return parser.parse_args()


def main():
    """
    Main function that executes the specified task based on the command-line arguments.
    """
    args = create_args()
    if args.task == "train":
        os.environ["WANDB_PROJECT"] = args.project_name
        if not args.wandb:
            os.environ["WANDB_API_KEY"] = args.wandb_key

        if args.search:
            print("Warning. Hyperparameter search does not work on distributed setups")
            assert args.fresh_start
        else:
            os.environ["WANDB_LOG_MODEL"] = "checkpoint"

        training_loop.train(args)
    else:
        raise ValueError("Task not recognized")


if __name__ == "__main__":
    main()
