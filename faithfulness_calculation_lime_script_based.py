import argparse
from pprint import pprint

import evaluate
import numpy as np
import pandas as pd
import torch
from torch import nn
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    EvalPrediction,
    OPTForSequenceClassification,
    Trainer,
    TrainingArguments,
)
import wandb

MODEL = "facebook/opt-350m"
MAX_POSITION_EMBEDDINGS = 2048

from dataclasses import dataclass

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

CHECKPOINT_DIR = "OPT-350m-mimic-full"
VAL_DATASET_PATH = "data/val_9.csv"
CODE_PATH = "data/icd9_codes.csv"

# Load dataset
tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=True, device=device)

code_labels = pd.read_csv("data/icd9_codes.csv")
dataset = load_dataset("csv", data_files=VAL_DATASET_PATH)

# Create class dictionaries
classes = [class_ for class_ in code_labels["icd_code"] if class_]
class2id = {class_: id for id, class_ in enumerate(classes)}
id2class = {id: class_ for class_, id in class2id.items()}


def multi_labels_to_ids(labels: list[str]) -> list[float]:
    ids = [0.0] * len(class2id)  # BCELoss requires float as target type
    for label in labels:
        ids[class2id[label]] = 1.0
    return ids


def preprocess_function(example):
    result = tokenizer(
        example["text"], truncation=True, padding=True, max_length=MAX_POSITION_EMBEDDINGS
    )
    result["labels"] = [multi_labels_to_ids(eval(label)) for label in example["labels"]]
    return result

config, unused_kwargs = AutoConfig.from_pretrained(
    MODEL,
    num_labels=len(classes),
    id2label=id2class,
    label2id=class2id,
    problem_type="multi_label_classification",
    return_unused_kwargs=True,
)

if unused_kwargs:
    print(f"Unused kwargs: {unused_kwargs}")

model = OPTForSequenceClassification.from_pretrained(
    MODEL,
    config=config,
    torch_dtype=torch.float16,
).to(device)

model.load_adapter(CHECKPOINT_DIR)

untokenized_dataset = load_dataset("csv", data_files=VAL_DATASET_PATH)

# print(untokenized_dataset['train'][0])

import lime
from lime import lime_text
from lime.lime_text import LimeTextExplainer
from lime.lime_text import IndexedString
import numpy as np
import torch.nn.functional as F
from time import time


explainer = LimeTextExplainer(class_names=classes, bow=False)

def predictor_opt(texts):
    # print(len(texts))
    tk = tokenizer(texts, return_tensors="pt",truncation=True, padding=True, max_length=MAX_POSITION_EMBEDDINGS).to(device)
    outputs = model(**tk)
    tensor_logits = outputs[0]
    probas = tensor_logits.sigmoid().detach().cpu().numpy()
    del tk, tensor_logits
    # probas = F.sigmoid(tensor_logits).detach().cpu().numpy()
    return probas

# used by the faithfulness function
def predictor_model(texts, model, tokenizer):
    # print(len(texts))
    tk = tokenizer(texts, return_tensors="pt",truncation=True, padding=True, max_length=MAX_POSITION_EMBEDDINGS).to(device)
    outputs = model(**tk)
    tensor_logits = outputs[0]
    probas = tensor_logits.sigmoid().detach().cpu().numpy()
    del tk, tensor_logits
    # probas = F.sigmoid(tensor_logits).detach().cpu().numpy()
    return probas


# create a test with 10 instances for faithfulness evaluation
# from transformers import AutoTokenizer

# get the lime evaluations of each instance
import faithfulness_shap_utils
# this reimports the library for easy testing in the notebook
import importlib
importlib.reload(faithfulness_shap_utils)

samples_start = 0
samples_end = 3

instances = untokenized_dataset["train"][samples_start:samples_end]["text"]
# print(len(instances))

# print(instances)
explainer = LimeTextExplainer(class_names=classes, bow=False)

indexed_text, index_array_rationalle = faithfulness_shap_utils.lime_create_index_arrays(instances, predictor_opt, explainer)

# faithfulness.save_indexed_strs(indexed_text,  index_array_rationalle, "test.npz")
# indexed_text, index_array_rationalle = faithfulness.load_indexed_strs("test.npz")

# print(indexed_text)
# print(index_array_rationalle)

# # remove the rationale words
rationalle_removed = faithfulness_shap_utils.remove_rationale_words(indexed_text, index_array_rationalle)
others_removed = faithfulness_shap_utils.remove_other_words(indexed_text, index_array_rationalle)

# rationalle_removed = rationalle_removed + rationalle_removed + rationalle_removed + rationalle_removed + rationalle_removed
# others_removed = others_removed + others_removed + others_removed + others_removed + others_removed 
# instances = instances + instances + instances + instances + instances

# print(rationalle_removed)

# print(len(rationalle_removed))
# print(len(others_removed))

# the extra list is needed since the function expects a list of instances each coming from a different interpretability method
# testing multi input by duplicating the arrays, don't actually do this
ind, faith = faithfulness_shap_utils.calculate_faithfulness(instances, [rationalle_removed, rationalle_removed], [others_removed, others_removed], model, tokenizer, predictor_model)
# print(ind)
# print(faith)
