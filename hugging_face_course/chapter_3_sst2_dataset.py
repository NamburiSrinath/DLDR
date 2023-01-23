### Trying this data-laoding part on GLUE SST-2 dataset which doesn't have pair of sentences

from datasets import load_dataset

raw_dataset = load_dataset("glue", "sst2")
# This has only one sentence and a corresponding label
print(raw_dataset)

from transformers import AutoTokenizer

# As this is a sentiment-classification task, we have to use Encoder-based model
# Currently using distilbert-base-uncased

# checkpoint = 'distilbert-base-uncased'
checkpoint = 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# Apply padding during batching i.e Dynamic padding
def tokenize_function(example):
    # return tokenizer(
    #     example['sentence'], truncation=True, padding='max_length', max_length=128
    # )

    return tokenizer(
        example['sentence'], truncation=True
    )

tokenize_dataset = raw_dataset.map(tokenize_function, batched=True)
print(tokenize_dataset.column_names)
print("-------")

# We have 5 columns - 'sentence', 'label', 'idx', 'input_ids', 'attention_mask
# We need to remove few columns before passing it to the dynamic padding function.

tokenize_dataset = tokenize_dataset.remove_columns(['sentence', 'idx'])
# tokenize_dataset = tokenize_dataset.rename_column('label', 'labels')
tokenize_dataset = tokenize_dataset.with_format('torch')
print(tokenize_dataset.column_names)
print("-------")

from transformers import DataCollatorWithPadding
from torch.utils.data import DataLoader


data_collator = DataCollatorWithPadding(tokenizer)
train_dataloader = DataLoader(tokenize_dataset["train"], batch_size=16, shuffle=True, collate_fn=data_collator)

from transformers import AutoModelForSequenceClassification
import torch
from transformers import AdamW

model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
# All the model parameters are to be optimized
optimizer = AdamW(model.parameters())

# Well this is not the best way to train and is not optimized as well
for step, batch in enumerate(train_dataloader):
    print(batch["input_ids"].shape)
    loss = model(**batch).loss
    print(loss)
    loss.backward()
    optimizer.step()
    if step > 5:
        break



####### A proper way to finetune SST-2 dataset using trainer-api...similar to the other one,
# Only change is at the tokenizer part where we need to send only one sentence
