# please note that is code is based on https://huggingface.co/docs/transformers/en/training
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
import numpy as np
import evaluate
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from transformers import AutoTokenizer,  OPTForSequenceClassification, TrainingArguments, EvalPrediction
from tqdm.auto import tqdm
from MIMICDataset import MimicDataset
from transformers import TrainingArguments, Trainer
import pandas as pd


def tokenize_dataset(dataset):
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-6.7b")
    
    def tokenize_function(dataset):
        return tokenizer(dataset["text"], padding="max_length", truncation=True)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    return tokenized_datasets


def calculate_accuracy(predictions, labels):
    return np.mean(predictions == labels)


def training_loop(dataset_code, dataset_label, icd_code, num_epochs=50, batch_size = 8):
    dataset = MimicDataset(dataset_code, dataset_label)
    tokenized_datasets = tokenize_dataset(dataset)
    icd_labels = pd.read_csv(dataset_label)
    
    # get number of icd codes from length of the dataset_label file
    code_count = len(icd_labels)

    # for testing
    small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
    small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))

    model = OPTForSequenceClassification.from_pretrained("ArthurZ/opt-350m-dummy-sc", num_labels=code_count, problem_type="multi_label_classification")
    
    # metrics are incorrect right now, i will add a custom metric for multi-label classification
    metric_name = "f1"

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)
    

    small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
    small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))

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
        f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
        roc_auc = roc_auc_score(y_true, y_pred, average = 'micro')
        accuracy = accuracy_score(y_true, y_pred)
        # return as dictionary
        metrics = {'f1': f1_micro_average,
                'roc_auc': roc_auc,
                'accuracy': accuracy}
        return metrics

    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, 
                tuple) else p.predictions
        result = multi_label_metrics(
            predictions=preds, 
            labels=p.label_ids)
        return result
    
    training_args = TrainingArguments(f"opt-finetuned-{icd_code}",
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model=metric_name)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=small_train_dataset,
        eval_dataset=small_eval_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    
    
if __name__ == "__main__":
    icd_code = "icd10"
    dataset_codes = f"data/{icd_code}_codes.csv"
    dataset_labels = f"data/mimic-iv-{icd_code}.csv"
    training_loop(dataset_codes, dataset_labels, icd_code)