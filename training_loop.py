# please note that is code is based on https://huggingface.co/docs/transformers/en/training
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
import numpy as np
import evaluate
from transformers import AutoTokenizer,  AutoModelForSequenceClassification, TrainingArguments, get_scheduler
from tqdm.auto import tqdm
from MIMICDataset import MimicDataset


def tokenize_dataset(dataset):
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-6.7b")
    
    def tokenize_function(dataset):
        return tokenizer(dataset["text"], padding="max_length", truncation=True)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    return tokenized_datasets


def calculate_accuracy(predictions, labels):
    return np.mean(predictions == labels)


def training_loop(num_epochs):
    dataset = MimicDataset("mimic-iv-icd9.csv")
    tokenized_datasets = tokenize_dataset(dataset)

    # for testing
    small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
    small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))

    model = AutoModelForSequenceClassification.from_pretrained(
        "facebook/opt-6.7b",
        num_labels=5
    )
        
    metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)
    

    small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
    small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))


    train_dataloader = DataLoader(small_train_dataset, shuffle=True, batch_size=8)
    eval_dataloader = DataLoader(small_eval_dataset, batch_size=8)

    # setup model and optimizer
    optimizer = AdamW(model.parameters(), lr=5e-5)
    
    for param in model.parameters():
        param.requires_grad = False  # freeze the model - train adapters later
        if param.ndim == 1:
            # cast the small parameters (e.g. layernorm) to fp32 for stability
            param.data = param.data.to(torch.float32)

    model.gradient_checkpointing_enable()  # reduce number of stored activations
    model.enable_input_require_grads()

    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )
    
    criterion = torch.nn.CrossEntropyLoss()

    # send the model to the gpu if available
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    # create a training progress bar
    progress_bar = tqdm(range(num_training_steps))
    
    # evaluation lists
    iters, train_losses, train_acc, val_acc = [], [], [], []
    iter_num = 0

    # training loop
    model.train()
    for epoch in range(num_epochs):
        for data, labels in iter(train_dataloader):
            # this code may be incorrect, I need to test it tomorrow            
            # batch = {k: v.to(device) for k, v in batch.items()}
            predictions = model(data)
            loss = criterion(predictions, labels)
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
                
            # test loop - check model accuracy
            iter_num += 1
            if iter_num % 10 == 0:
                loss = float(loss)
                train_losses.append(loss)
                
                train_acc = calculate_accuracy(predictions, labels)
                iters.append(iter_num)
                train_acc.append(train_acc)
                val_acc.append(val_acc)
                

    # evaluation loop
    # we might not need this loop, I'll test it tomorrow (3/27)
    metric = evaluate.load("accuracy")
    model.eval()
    for batch in eval_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])

    metric.compute()
    
    
if __name__ == "__main__":
    training_loop(10)