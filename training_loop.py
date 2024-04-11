# please note that is code is based on https://huggingface.co/docs/transformers/en/training
import argparse
from pprint import pprint

import evaluate
import numpy as np
import pandas as pd
import torch
import wandb
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from transformers import (
    AutoTokenizer,
    AutoConfig,
    EvalPrediction,
    OPTForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)

MODEL = "facebook/opt-350m"

def train(args: argparse.Namespace):
    # for some reason the trainer has issues passing parameters to the model_init function so this variable needs to be global
    global tokenizer
    global classes
    global class2id
    global id2class
    global clf_metrics

    global sigmoid

    print("Loading datasets")
    data_files = {
        "train": args.train_path,
        "validation": args.val_path,
        "test": args.test_path,
    }

    code_labels = pd.read_csv(args.code_labels)
    """train_set = pd.read_csv(args.train_path)
    val_set = pd.read_csv(args.val_path)
    test_set = pd.read_csv(args.test_path)"""

    dataset = load_dataset("csv", data_files=data_files, cache_dir=args.cache_dir)

    # Create class dictionaries
    classes = [class_ for class_ in code_labels["icd_code"] if class_]
    class2id = {class_: id for id, class_ in enumerate(classes)}
    id2class = {id: class_ for class_, id in class2id.items()}

    clf_metrics = evaluate.combine(["accuracy", "f1", "precision", "recall", "roc_auc"])

    print("Tokenizing datasets. Loading from cache if available.")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=True)

    torch.cuda.set_device(0)
    torch.cuda.current_device()
    sigmoid = torch.nn.Sigmoid()

    # Run dummy tokenization run first to circumvent bug with hashing changing
    # tokenizer("Some", "test")

    dataset = dataset.map(tokenize, load_from_cache_file=True, batched=True, num_proc=8)

    # make sure to update num_train_epochs for actual training
    # note - save stratedy and evaluation strategy need to match
    # change batch sizes back to 8 for testing
    training_args = TrainingArguments(
        output_dir=args.checkpoint_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_steps=args.save_interval,
        learning_rate=2e-5,
        num_train_epochs=1,
        weight_decay=0.01,
        load_best_model_at_end=True,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    pprint(f"num_labels: {len(code_labels)}")

    if args.wandb_key:
        pprint("Using wandb")
        wandb.login(key=args.wandb_key)
        training_args.report_to = ["wandb"]

    if args.fresh_start:
        pprint("Fresh Start")
        trainer = hyperparameter_search(
            model_init,
            training_args,
            dataset,
            tokenizer,
            compute_metrics,
            data_collator,
            n_trials=2,
        )

    else:
        # Load model from local checkpoint
        pprint("Attempting to load local model checkpoint")
        model = OPTForSequenceClassification.from_pretrained(args.checkpoint_dir)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"],
            tokenizer=tokenizer,
            data_collator=data_collator,
        )

        trainer.train(resume_from_checkpoint=True)

    trainer.save_model(args.checkpoint_dir)
    trainer.save_state()


def multi_labels_to_ids(labels: list[str]) -> list[float]:
    ids = [0.0] * len(class2id)  # BCELoss requires float as target type
    for label in labels:
        ids[class2id[label]] = 1.0
    return ids


def tokenize(example):
    result = tokenizer(
        example["text"],
        add_special_tokens=True,
    )
    result["label"] = [multi_labels_to_ids(eval(label)) for label in example["label"]]

    return result


def model_init():
    """Model init for use for hyperparameter_search"""
    return OPTForSequenceClassification.from_pretrained(
        MODEL,
        num_labels=len(classes),
        id2label=id2class,
        label2id=class2id,
        problem_type="multi_label_classification",
    )


def wandb_hp_space(trial):
    """
    Returns a dictionary representing the hyperparameter space for Weights & Biases (wandb) optimization.

    Args:
        trial: An object representing the current optimization trial.

    Returns:
        A dictionary containing the hyperparameter space for wandb optimization. The dictionary has the following structure:
        {
            "method": "random",
            "metric": {"name": "loss", "goal": "minimize"},
            "parameters": {
                "learning_rate": {"distribution": "uniform", "min": 1e-6, "max": 1e-4},
                "per_device_train_batch_size": {"values": [16, 32, 64]},
            },
        }
    """
    return {
        "method": "random",
        "metric": {"name": "loss", "goal": "minimize"},
        "parameters": {
            "learning_rate": {"distribution": "uniform", "min": 1e-6, "max": 1e-4},
            "per_device_train_batch_size": {"values": [16, 32, 64]},
        },
    }


def hyperparameter_search(
    model_init,
    args,
    dataset,
    tokenizer,
    compute_metrics,
    data_collator,
    n_trials: int = 10,
):
    """
    Perform hyperparameter search using the Trainer class.

    Args:
        model_init (function): A function that initializes the model.
        num_labels (int): The number of labels in the dataset.
        args: The arguments for training the model.
        train_dataset: The training dataset.
        eval_dataset: The evaluation dataset.
        tokenizer: The tokenizer used for tokenizing the input data.
        compute_metrics: A function that computes evaluation metrics.
        n_trials (int, optional): The number of hyperparameter search trials. Defaults to 10.

    Returns:
        Trainer: Trainer with attributes of best run

    """
    pprint(f"Doing hyperparameter search with {n_trials} trials")

    trainer = Trainer(
        model_init=model_init,
        args=args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
    )

    best_run = trainer.hyperparameter_search(
        hp_space=wandb_hp_space,
        n_trials=n_trials,
        direction="minimize",
        backend="wandb",
    )

    pprint(best_run)

    for n, v in best_run.hyperparameters.items():
        setattr(trainer.args, n, v)
    return trainer


def multi_label_metrics(predictions, labels, threshold=0.5):
    # source: https://jesusleal.io/2021/04/21/Longformer-multilabel-classification/
    # first, apply sigmoid on predictions which are of shape (batch_size, num_labels)
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))
    # next, use threshold to turn them into integer predictions
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1
    # finally, compute metrics
    y_true = labels
    f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average="micro")
    roc_auc = roc_auc_score(y_true, y_pred, average="micro")
    accuracy = accuracy_score(y_true, y_pred)
    # return as dictionary
    metrics = {"f1": f1_micro_average, "roc_auc": roc_auc, "accuracy": accuracy}
    return metrics


def compute_metrics(p: EvalPrediction):
    """preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    result = multi_label_metrics(predictions=preds, labels=p.label_ids)
    return result"""

    predictions, labels = p
    predictions = sigmoid(predictions)
    predictions = (predictions > 0.5).astype(int).reshape(-1)
    return clf_metrics.compute(
        predictions=predictions, references=labels.astype(int).reshape(-1)
    )


# TODO: leaving this code here for now but we should remove it before final submission
def training_loop(
    model,
    dataset: pd.DataFrame,
    dataset_codes: pd.DataFrame,
    icd_code,
    training_args: TrainingArguments,
    trainer: Trainer,
):
    # split in to training and validation datasets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    # send the model to the gpu if available
    device = (
        torch.cuda_set_device(1) if torch.cuda.is_available() else torch.device("cpu")
    )
    model.to(device)

    training_args = TrainingArguments(output_dir="test_trainer")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()


if __name__ == "__main__":
    icd_code = "icd10"
    dataset_codes = f"data/{icd_code}_codes.csv"
    dataset = f"data/mimic-iv-{icd_code}-small.csv"
    # dataset_labels = f"data/mimic-iv-{icd_code}.csv"
    training_loop(dataset, dataset_codes, icd_code)
