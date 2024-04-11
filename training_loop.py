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
    EvalPrediction,
    OPTForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)

MODEL = "facebook/opt-1.3b"


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

    sigmoid = torch.nn.Sigmoid()

    # Run dummy tokenization run first to circumvent bug with hashing changing
    # tokenizer("Some", "test")

    dataset = dataset.map(tokenize, load_from_cache_file=True, batched=True, num_proc=8)

    # make sure to update num_train_epochs for actual training
    # note - save stratedy and evaluation strategy need to match
    # change batch sizes back to 8 for testing
    training_args = TrainingArguments(
        output_dir=args.checkpoint_dir,
        dataloader_num_workers=3,
        evaluation_strategy="steps",
        eval_steps=args.save_interval,
        save_strategy="steps",
        accelerator_config={"split_batches": True},
        save_steps=args.save_interval,
        learning_rate=2e-5,
        num_train_epochs=2,
        weight_decay=0.01,
        load_best_model_at_end=True,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adafactor",
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    pprint(f"num_labels: {len(code_labels)}")

    if args.wandb_key:
        pprint("Using wandb")
        wandb.login(key=args.wandb_key)
        training_args.report_to = ["wandb"]

    if args.fresh_start:
        pprint("Fresh Start")

        trainer = Trainer(
            model_init=model_init,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"],
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
            data_collator=data_collator,
        )
        trainer.train()
    else:
        # Load model from local checkpoint
        pprint("Attempting to load local model checkpoint")
        model = OPTForSequenceClassification.from_pretrained(
            args.checkpoint_dir,
            torch_dtype=torch.float16,
            attn_implementation="flash_attention_2",
        )

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
    result = tokenizer(example["text"], truncation=True, max_length=2048)
    result["label"] = [multi_labels_to_ids(eval(label)) for label in example["label"]]

    return result


def model_init():
    """Model init for use for hyperparameter_search"""
    model = OPTForSequenceClassification.from_pretrained(
        MODEL,
        num_labels=len(classes),
        id2label=id2class,
        label2id=class2id,
        problem_type="multi_label_classification",
        torch_dtype=torch.float16,
        use_cache=False,
        attn_implementation="flash_attention_2",
    )

    model.to("cuda")

    return model


@DeprecationWarning
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
    return result"""

    predictions, labels = p
    predictions = sigmoid(predictions)
    predictions = (predictions > 0.5).astype(int).reshape(-1)
    return clf_metrics.compute(
        predictions=predictions, references=labels.astype(int).reshape(-1)
    )


if __name__ == "__main__":
    print("Please use manager.py")
