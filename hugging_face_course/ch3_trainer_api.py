"""
Trainer API to finetune a model
"""
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding

raw_datasets = load_dataset("glue", "mrpc")
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)


tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

from transformers import TrainingArguments
# This will save all the hyperparameters in the given folder
# Can pass learning rate, no of epochs etc; Basically hyperparameters initialization
training_arguments = TrainingArguments("testing-trainer")

# Sequence classification model with labels as 0, 1

from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

from transformers import Trainer

# Pass model, train/validate datasets, arguments, padding and tokenizer
# We are not doing the removing, renaming columns etc; which we did in preprocessing dataset
# Trainer will take care of that

trainer = Trainer(
    model,
    training_arguments,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['validation'],
    data_collator=data_collator,
    tokenizer=tokenizer
)

# This is training
trainer.train()

# Can use evaluation_strategy as steps or epochs to do evaluation
# Also need to initialize the metrics for evaluation (accuracy etc;) basically other than loss

# First get the predictions

# predict() will give predictions, label_ids and metrics which has the loss by default
predictions =trainer.predict(tokenized_datasets["validation"])

# Predictions are logits, label_ids are ground truths
print(predictions.predictions.shape, predictions.label_ids.shape)

# Change logits to labels
import numpy as np
# Just get the index of the max logit
preds = np.argmax(predictions.predictions, axis=-1)

# Load the evaluation metrics
import evaluate
metric = evaluate.load("glue", "mrpc")
# Pass the predictions and the ground truth labels
print(metric.compute(predictions=preds, references=predictions.label_ids))

# We can wrap this in a compute_metric function and use it in trainer

# eval_preds are nothing but predictions in the above example.
# Which is basically passing validation dataset to the predict function in trainer
def compute_metric(eval_preds):
    metric = evaluate.load("glue", "mrpc")
    # Get predicted logits and the ground truth from eval_preds
    logits, labels = eval_preds
    # Convert logits to prediction labels
    preds = np.argmax(logits, axis=-1)
    # Compute accuracy and F1 score for predictions and labels
    return metric.compute(preds, labels)

model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
training_arguments = TrainingArguments("testing-trainer", evaluation_strategy='epoch')

trainer = Trainer(
    model,
    training_arguments,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets['validation'],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metric=compute_metric
)

trainer.train()






