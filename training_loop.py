import argparse
from pprint import pprint

import evaluate
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoConfig,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    OPTForSequenceClassification,
    Trainer,
    TrainingArguments,
)
import wandb

MODEL = "facebook/opt-2.7b"
MAX_POSITION_EMBEDDINGS = 2560


def train(args: argparse.Namespace):
    # for some reason the trainer has issues passing parameters to the model_init function so this variable needs to be global
    global tokenizer
    global classes
    global class2id
    global id2class

    print("Loading datasets")
    data_files = {
        "train": args.train_path,
        "validation": args.val_path,
        "test": args.test_path,
    }

    code_labels = pd.read_csv(args.code_labels)
    dataset = load_dataset("csv", data_files=data_files, cache_dir=args.cache_dir)

    # Create class dictionaries
    classes = [class_ for class_ in code_labels["icd_code"] if class_]
    class2id = {class_: id for id, class_ in enumerate(classes)}
    id2class = {id: class_ for class_, id in class2id.items()}

    print("Tokenizing datasets. Loading from cache if available.")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=True)

    dataset = dataset.map(tokenize, load_from_cache_file=True, batched=True, num_proc=8)

    create_metrics(args)

    # note - save stratedy and evaluation strategy need to match
    training_args = TrainingArguments(
        auto_find_batch_size=True,
        disable_tqdm=args.disable_tqdm,
        output_dir=args.checkpoint_dir,
        dataloader_num_workers=3,
        evaluation_strategy="epoch",
        eval_steps=args.save_interval,
        save_strategy="epoch",
        save_steps=args.save_interval,
        warmup_steps=200,
        learning_rate=2e-6,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        ddp_find_unused_parameters=False,
        eval_accumulation_steps=250,
        logging_steps=200,
        # optim="adafactor",
        save_total_limit=4,
    )

    if args.tiny:
        # Use tiny subset of dataset
        dataset["train"] = dataset["train"].shard(index=1, num_shards=60)
        dataset["validation"] = dataset["validation"].shard(index=1, num_shards=60)
        dataset["test"] = dataset["test"].shard(index=1, num_shards=60)
        training_args.evaluation_strategy = "epoch"
        training_args.eval_steps = 1

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
        if args.search:
            best_run = trainer.hyperparameter_search(
                n_trials=10,
                backend="optuna",
                hp_space=optuna_hp_space,
                direction="maximize",
            )

            print(best_run)
        else:
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


def optuna_hp_space(trial):

    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 0, 0.1),
        "warmup_steps": trial.suggest_int("warmup_steps", 0, 400, step=200)
    }


def create_metrics(args):
    global clf_metrics

    print("Creating evaluation metrics")

    f1 = evaluate.load(
        "f1",
        config_name="multilabel",
    )
    precision = evaluate.load(
        "precision",
        config_name="multilabel",
    )
    recall = evaluate.load(
        "recall",
        config_name="multilabel",
    )

    clf_metrics = evaluate.combine([f1, precision, recall])


def multi_labels_to_ids(labels: list[str]) -> list[float]:
    ids = [0.0] * len(class2id)  # BCELoss requires float as target type
    for label in labels:
        ids[class2id[label]] = 1.0
    return ids


def tokenize(example):
    result = tokenizer(
        example["text"], truncation=True, max_length=MAX_POSITION_EMBEDDINGS
    )
    result["label"] = [multi_labels_to_ids(eval(label)) for label in example["label"]]

    return result


def model_init():
    """Model init for use for hyperparameter_search"""
    lora_config = LoraConfig(
        r=16,
        target_modules=["q_proj", "v_proj"],
        task_type=TaskType.SEQ_CLS,
        lora_alpha=32,
        lora_dropout=0.05,
    )

    config, unused_kwargs = AutoConfig.from_pretrained(
        MODEL,
        num_labels=len(classes),
        id2label=id2class,
        label2id=class2id,
        problem_type="multi_label_classification",
        attn_implementation="flash_attention_2",
        return_unused_kwargs=True,
        max_position_embeddings=MAX_POSITION_EMBEDDINGS,
    )

    if unused_kwargs:
        print(f"Unused kwargs: {unused_kwargs}")

    model = OPTForSequenceClassification.from_pretrained(
        MODEL,
        config=config,
        torch_dtype=torch.float16,
        ignore_mismatched_sizes=True,
    )
    model.tie_weights()
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model


def compute_metrics(p: EvalPrediction):
    """preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    return result"""
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions

    preds = np.where(preds > 0, 1, 0)

    result = clf_metrics.compute(
        predictions=preds, references=p.label_ids, average="micro"
    )
    return result


if __name__ == "__main__":
    print("Please use manager.py")
