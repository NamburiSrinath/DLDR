# """
# Using hugging face datasets to load and process
# """

# from datasets import load_dataset

# # Loading MRPC dataset from GLUE benchmark. Has 2 sentence pairs and need to tell if they both link or not
# raw_datasets = load_dataset("glue", "mrpc")

# # Returns a DatasetDict object which has train, validation and test data. 
# print(raw_datasets)
# raw_train_dataset = raw_datasets["train"]
# print(raw_train_dataset[0])
# print(raw_train_dataset.features)

# # label has the ground truth and is of type ClassType. More details can be found in names sub-field of labels

# from transformers import AutoTokenizer
# checkpoint = "bert-base-uncased"

# tokenizer = AutoTokenizer.from_pretrained(checkpoint)
# tokenized_sentences_1 = tokenizer(raw_train_dataset['sentence1'])
# tokenized_sentences_2 = tokenizer(raw_train_dataset['sentence2'])
# print(tokenized_sentences_1, tokenized_sentences_2)
# print("--------")
# # while this tokenized all the sentences, we need to pair them and send that to the network. 
# # Also, this is not the fastest way to tokenize

# # We can pass multiple sentences to tokenizer (this helps to construct map function)
# inputs = tokenizer("This is the first sentence.", "This is the second one.")
# print(inputs)
# print("--------")
# # token-type IDs are used to distinguish between sentences. 
# # Token-type IDs are returned only to specific models depending on their pretraining objectives
# print(tokenizer.convert_ids_to_tokens(inputs["input_ids"]))
# print("--------")

# tokenized_dataset = tokenizer(
#     raw_datasets["train"]["sentence1"],
#     raw_datasets["train"]["sentence2"],
#     padding=True,
#     truncation=True,
# )

# # This works perfectly well as long as we have enough RAM as the returned object is a Dictionary
# print(tokenized_dataset)
# print("--------")

# # But HuggingFace Datasets has implemented in Apache Arrow and saves lot of RAM etc; by loading everything
# # to HF Datasets

# # We can use dynamic padding instead of padding='max_length' to speed-up tokenization
# def tokenize_function(example):
#     return tokenizer(
#         example['sentence1'], example['sentence2'], truncation=True, max_length=128, padding='max_length'
#     )

# # batched helps to send multiple pairs of sentences together 
# tokenized_dataset = raw_datasets.map(tokenize_function, batched=True)
# print(tokenized_dataset.column_names)
# print("--------")

# # removed unnecessary columns, changed name of one column which the model expects and converted to torch
# tokenized_dataset = tokenized_dataset.remove_columns(['idx', 'sentence1', 'sentence2'])
# tokenized_dataset = tokenized_dataset.rename_column('label', 'labels')
# tokenized_dataset = tokenized_dataset.with_format('torch')
# print(tokenized_dataset['train'][:5])
# print("---------")

# # For quick checks, can select a smaller dataset
# small_dataset = tokenized_dataset['train'].select(range(10))
# print(small_dataset)
# print("-------")

# # Dynamic padding can be used to speed up padding. Padding during batch-creation
# from transformers import AutoTokenizer
# from torch.utils.data import DataLoader
# from transformers import DataCollatorWithPadding

# checkpoint = "bert-base-uncased"
# tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# # Here we removed the padding and max-length argument. We do padding during batch creation
# def tokenizer_function(example):
#     return tokenizer(
#         example["sentence1"], example["sentence2"], truncation=True
#     )
# tokenized_dataset = raw_datasets.map(tokenizer_function, batched=True)
# tokenized_dataset = tokenized_dataset.remove_columns(['idx', 'sentence1', 'sentence2'])
# tokenized_dataset = tokenized_dataset.rename_column('label', 'labels')
# tokenized_dataset = tokenized_dataset.with_format('torch')
# print(tokenized_dataset['train'])
# print("---------")

# data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
# # This collate function takes care of dynamic padding
# train_dataloader = DataLoader(tokenized_dataset['train'], batch_size=16, shuffle=True, collate_fn=data_collator)
# for step, batch in enumerate(train_dataloader):
#     print(batch['input_ids'].shape)
#     if step > 10:
#         break
