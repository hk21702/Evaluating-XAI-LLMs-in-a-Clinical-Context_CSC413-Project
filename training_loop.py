# please note that is code is based on https://huggingface.co/docs/transformers/en/training
import argparse
from pprint import pprint

import evaluate
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoTokenizer,
    AutoConfig,
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
    global clf_metrics

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

    clf_metrics = evaluate.combine(["accuracy", "f1", "precision", "recall"])

    print("Tokenizing datasets. Loading from cache if available.")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=True)

    dataset = dataset.map(tokenize, load_from_cache_file=True, batched=True, num_proc=8)

    # note - save stratedy and evaluation strategy need to match
    training_args = TrainingArguments(
        disable_tqdm=args.disable_tqdm,
        output_dir=args.checkpoint_dir,
        dataloader_num_workers=3,
        evaluation_strategy="steps",
        eval_steps=args.save_interval,
        save_strategy="steps",
        accelerator_config={"split_batches": True},
        save_steps=args.save_interval,
        learning_rate=2e-5,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        per_device_train_batch_size=8,
        gradient_checkpointing=True,
        ddp_find_unused_parameters=False,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adafactor",
        save_total_limit=4,
    )

    if args.tiny:
        # Use tiny subset of dataset
        dataset["train"] = dataset["train"].select(range(800))
        dataset["validation"] = dataset["validation"].select(range(50))
        dataset["test"] = dataset["test"].select(range(100))
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
        use_cache=False,  # Renable this for inference!
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

    predictions, labels = p
    predictions = 1 / (1 + np.exp(-predictions))
    predictions = (predictions > 0.5).astype(int).reshape(-1)
    return clf_metrics.compute(
        predictions=predictions, references=labels.astype(int).reshape(-1)
    )


if __name__ == "__main__":
    print("Please use manager.py")
