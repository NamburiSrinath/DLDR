"""
Finetune the model on SST-2 dataset without using Trainer API

MRPC dataset code is available in https://huggingface.co/course/chapter3/4?fw=pt
"""
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding

raw_dataset = load_dataset("glue", "sst2")
checkpoint = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

def tokenize_function(example):
    return tokenizer(
        example['sentence'], truncation=True
    )

tokenize_dataset = raw_dataset.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer)

# We have to remove unnecessary columns that is done by Trainer API under-the-hood
tokenize_dataset = tokenize_dataset.remove_columns(['sentence', 'idx'])
tokenize_dataset = tokenize_dataset.rename_column('label', 'labels')
tokenize_dataset = tokenize_dataset.with_format('torch')
print(tokenize_dataset.column_names)
print("--------")

from torch.utils.data import DataLoader

# Dynamic Padding
train_dataloader = DataLoader(tokenize_dataset["train"], shuffle=True, batch_size=32, collate_fn=data_collator)
validate_dataloader = DataLoader(tokenize_dataset['validation'], batch_size=32, collate_fn=data_collator)

for batch in train_dataloader:
    break
print({k: v.shape for k, v in batch.items()})
print("------------")

# DATA PREPROCESSING DONE

from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

outputs = model(**batch)
print(outputs.loss, outputs.logits.shape)
print("---------")

from transformers import AdamW
optimizer = AdamW(model.parameters(), lr=5e-5)

# Learning rate scheduler to reduce learning rate as the steps progress
# Implemented inside the Trainer API. 

from transformers import get_scheduler

num_epochs=3
num_training_steps = num_epochs*len(train_dataloader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps
)

import torch

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)
print(device)
print("---------")

# TRAINING LOOP 
from tqdm.auto import tqdm
progress_bar = tqdm(range(num_training_steps))

# Put in training mode so Dropout etc; will also be activated
model.train()
for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        # Compute loss from the present outputs
        loss = outputs.loss
        # Backprop the loss
        loss.backward()
        # Update the parameters using optimizer
        optimizer.step()
        # Update the learning rate as well
        lr_scheduler.step()
        # Zero out the optimizer grads, else the grads will keep on accumulating
        optimizer.zero_grad()
        progress_bar.update(1)

# EVALUATION LOOP
import evaluate
metric = evaluate.load("glue", "sst2")
# Put in evaluation mode
model.eval()
for batch in validate_dataloader:
    batch = {k: v.to(device) for k,v in batch.items()}
    # Don't do any backprop
    with torch.no_grad():
        outputs = model(**batch)
    
    # Transformer outputs are always logits
    logits = outputs.logits
    # Convert logits to prediction labels
    predictions = torch.argmax(logits, axis=-1)
    metric.add_batch(predictions=predictions, references=batch["labels"])

print(metric.compute())

