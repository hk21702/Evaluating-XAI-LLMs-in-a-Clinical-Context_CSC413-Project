# please note that is code is based on https://huggingface.co/docs/transformers/en/training
import argparse

import evaluate
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoTokenizer,
    EvalPrediction,
    OPTForSequenceClassification,
    Trainer,
    TrainingArguments,
)

from MIMICDataset import MimicDataset


def train(args: argparse.Namespace):
    pass


def training_loop(dataset, dataset_codes, icd_code, num_epochs=1, batch_size=1):
    mimic_dataset = pd.read_csv(dataset)
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-6.7b")

    # tokenize the dataset
    tokenized_dataset = mimic_dataset.apply(
        lambda x: tokenizer.encode_plus(
            x["text"],
            add_special_tokens=True,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ),
        axis=1,
    )
    print(tokenized_dataset[0])

    # extract the labels column
    labels = mimic_dataset["icd_proc"]

    dataset = MimicDataset(tokenized_dataset, labels, dataset_codes)

    # print(tokenized_dataset.items())
    # print(dataset[0])

    # split in to training and validation datasets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    # create data loaders
    icd_labels = pd.read_csv(dataset_codes)

    # print(dataset[0])
    # print(icd_labels)

    # get number of icd codes from length of the dataset_label file
    code_count = len(icd_labels)

    model = OPTForSequenceClassification.from_pretrained(
        "facebook/opt-350m",
        num_labels=code_count,
        problem_type="multi_label_classification",
    )

    # metrics are incorrect right now, i will add a custom metric for multi-label classification
    metric_name = "f1"

    # send the model to the gpu if available
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    training_args = TrainingArguments(output_dir="test_trainer")

    # source: https://jesusleal.io/2021/04/21/Longformer-multilabel-classification/
    def multi_label_metrics(predictions, labels, threshold=0.5):
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
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        result = multi_label_metrics(predictions=preds, labels=p.label_ids)
        return result

    training_args = TrainingArguments(
        f"opt-finetuned-{icd_code}",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model=metric_name,
    )

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
