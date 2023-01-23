"""
Similar to ch3_finetune_pytorch.py
In case we want to run using multiple GPUs and distributed training, we need to use accelerate API

Optional - Study more on Distributed training

Use accelerate config to create yaml file and accelerate launch xxx.py to start run 
"""
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding

raw_datasets = load_dataset("glue", "sst2")
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


def tokenize_function(example):
    return tokenizer(example["sentence1"], truncation=True)


tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

tokenized_datasets = tokenized_datasets.remove_columns(["sentence", "idx"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")
print(tokenized_datasets["train"].column_names)

from torch.utils.data import DataLoader

train_dataloader = DataLoader(
    tokenized_datasets["train"], shuffle=True, batch_size=8, collate_fn=data_collator
)
eval_dataloader = DataLoader(
    tokenized_datasets["validation"], batch_size=8, collate_fn=data_collator
)

# Importing accelerator API and instantiating the object
from accelerate import Accelerator
from transformers import AdamW, AutoModelForSequenceClassification, get_scheduler

accelerator = Accelerator()

model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
optimizer = AdamW(model.parameters(), lr=3e-5)

# prepare the train, evaluation dataloaders, model and optimizer which will be distributed across devices
train_dl, eval_dl, model, optimizer = accelerator.prepare(
    train_dataloader, eval_dataloader, model, optimizer
)

num_epochs = 3
num_training_steps = num_epochs * len(train_dl)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

progress_bar = tqdm(range(num_training_steps))

model.train()
for epoch in range(num_epochs):
    for batch in train_dl:
        # No need to load batch to device as it's taken care by prepare method
        outputs = model(**batch)
        loss = outputs.loss
        # Instead of loss.backward()
        accelerator.backward(loss)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

# Evaluation loop

from datasets import load_metric

metric = load_metric("glue", "sst2")

model.eval()
for batch in eval_dl:
    with torch.no_grad:
        outputs = model(**batch)

    logits = outputs.logits
    predictions = torch.argmax(logits, axis=-1)
    metric.add_batch(
        predictions=accelerator.gather(predictions), references=accelerator.gather(batch["labels"])
    )
print(metric.compute())

    

