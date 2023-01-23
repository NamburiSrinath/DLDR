# From module 2 of hugging face transformers

"""
Pipeline has 3 parts inside
1. Tokenizer (Tokenization)
2. Model
3. Post Processing 

1. AutoTokenizer
We can get tokenizer for a particular model
Text -> Tokens -> Special tokens appended (CLS, SEP) -> Token IDs

Attention masks is same as source attention. No need to care about padded ones, so this mask is present

2. IMPORTANT - AutoModel
It loads a model without it's pretraining head

If we want to use for a specific task, use AutoModelForSequenceClassfication
More general version is AutoModelForXXXX
Eg: *Model (retrieve the hidden states) - Like a feature extractor
*ForCausalLM
*ForMaskedLM
*ForMultipleChoice
*ForQuestionAnswering
*ForSequenceClassification
*ForTokenClassification

They usually return the logits, not the probabilities

3. PostProcessing
Convert logits to probabilities (softmax in Pytorch)

Notes:
Preprocessing for our task and the task which the model is trained on should be the same
"""

from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification

model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)

raw_inputs = [
    "I've been waiting for a HuggingFace course my whole life.",
    "I hate this so much!",
]
inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")
print(inputs)
print("----------")

model = AutoModelForSequenceClassification.from_pretrained(model_name)
output = model(**inputs)
print(output.logits.shape)
print("----------")

import torch
predictions = torch.nn.functional.softmax(output.logits, dim=-1)
print(predictions)
print("--------")
print(model.config.id2label)

# HF models output logits...outputs.logits. Need to convert to softmax on our own

from transformers import BertConfig, BertModel

# Loading a random initialized model from config
# Can also use AutoConfig 
config = BertConfig()
model = BertModel(config)

"""
Tokenization
1. Word based
    Has lots of representations for similar meaning words
    Complicates the size of the models
    Limit the vocabulary size by UNK, but different meaning words can be mapped to UNK if they aren't present in vocab
    This results in loss of information
    Thumb rule: Make sure you won't get many UNKs in Tokenizer
2. Character based
    Pros: Very small vocabulary, compared to words!!
    Cons: Can't capture the meaning/context!! Need complicated models to understand the interactions btw characters
    Will have large length input for model
3. Sub-word based
    Frequently occured words won't get splitted. 
    Non-frequently used words will be splitted to frequently used words instead of UNKs
    Eg: Dog -> Dog while Dogs -> Dog, ##s (## tells it's a part of a word. ## is from BERT).
    Eg algorithms: 
        WordNet - BERT, DistillBERT
        Unigram - ALBERT, XLNet - MultiLingual models
        Byte-Pair Encoding - GPT2, RoBERTa
"""

from transformers import AutoTokenizer

# Loading BERT based tokenizer using AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
print(tokenizer("Using a Transformer network is simple"))
print("----------")
# This generates Input IDs, token_type_ids and attention_masks
# Inside tokenizer pipeline!

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
sentence = "Hello Srinath, This is Siddharth! How are you?"

# Converts the sentence to tokens based on Tokenization algorithm used in the model. Here it's WordNet
sentence_to_tokens = tokenizer.tokenize(sentence)
print(sentence_to_tokens)
print("----------")

# This converts tokens to IDs, but we need to add start and end token ids to this before passing to model
tokens_to_ids = tokenizer.convert_tokens_to_ids(sentence_to_tokens)
print(tokens_to_ids)
print("-----------")

# This step adds start and end token ids
final_input_ids = tokenizer.prepare_for_model(tokens_to_ids)
print(final_input_ids)

# To decode the token ids i.e how the model sees the input
decoded_sentence = tokenizer.decode(final_input_ids['input_ids'])
print(decoded_sentence)
print("--------")

"""
Handling multiple sequences
    By default, the tokenizer needs multiple sentences!
"""
# This will fail
# import torch
# from transformers import AutoTokenizer, AutoModelForSequenceClassification

# checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
# tokenizer = AutoTokenizer.from_pretrained(checkpoint)
# model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

# sequence = "I've been waiting for a HuggingFace course my whole life."

# tokens = tokenizer.tokenize(sequence)
# ids = tokenizer.convert_tokens_to_ids(tokens)
# # We are sending only one sentence, but it expects a list of lists by default!
# input_ids = torch.tensor(ids)
# # This line will fail.
# model(input_ids)

# Instead, convert the single sentence ids to a list.
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

sequence = "I've been waiting for a HuggingFace course my whole life."

tokens = tokenizer.tokenize(sequence)
ids = tokenizer.convert_tokens_to_ids(tokens)
# We are sending only one sentence, but now have converted to list of lists
input_ids = torch.tensor([ids])
# This line pass
print(model(input_ids))
print("---------")

# We need to convert token ids list to a tensor and it can be done only if the length of each internal list
# stays the same. We can achieve it by padding!

model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

# Each sequence id is already a list of lists, so can be converted to tensor without an issue
sequence1_ids = [[200, 200, 200]]
sequence2_ids = [[200, 200]]

# Each tokenizer has a separate pad token id, can be called by a variable name
batched_ids = [
    [200, 200, 200],
    [200, 200, tokenizer.pad_token_id],
]

# 2, 3 has very different values and the reason is the model considered padding as well
# We need to use attention masks to tell the model to ignore stuff i.e don't pay attention on stuff that's 
# padded. This way, we can add padding to pass as input while can ignore by not paying attention
print(model(torch.tensor(sequence1_ids)).logits)
print(model(torch.tensor(sequence2_ids)).logits)
print(model(torch.tensor(batched_ids)).logits)
print("-----------")

# Batch + Padding + Attention masks
batched_ids = [
    [200, 200, 200],
    [200, 200, tokenizer.pad_token_id],
]
attention_mask = [
    [1, 1, 1],
    [1, 1, 0],
]

outputs = model(torch.tensor(batched_ids), attention_mask=torch.tensor(attention_mask))
print(outputs.logits)
print("-----------")

# AN example using a batch of sentences

sentences = [
    "I’ve been waiting for a HuggingFace course my whole life.", 
    "I hate this so much!"
    ]
# Passing the sentences to tokenizer will take care of everything
# Sentences -> Tokens -> IDs -> Adding special IDS -> Batching -> Padding and attention masking -> Model -> Output

inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
outputs = model(**inputs)
print(outputs.logits)
print("----------")

# Doing the above steps manually!!!
sentence1_to_tokens = tokenizer.tokenize(sentences[0])
sentence2_to_tokens = tokenizer.tokenize(sentences[1])
print(sentence1_to_tokens)
print(sentence2_to_tokens)
print("------------")

tokens1_to_ids = tokenizer.convert_tokens_to_ids(sentence1_to_tokens)
tokens2_to_ids = tokenizer.convert_tokens_to_ids(sentence2_to_tokens)
print(tokens1_to_ids)
print(tokens2_to_ids)
print("----------")

final_input1 = tokenizer.prepare_for_model(tokens1_to_ids)
final_input2 = tokenizer.prepare_for_model(tokens2_to_ids)
print(final_input1)
print(final_input2)
print("--------")

padding_array = [tokenizer.pad_token_id]*8
print(f"Padding array is  {padding_array}")
final_input2['input_ids'].extend(padding_array)

batched_ids = [
    final_input1['input_ids'],
    final_input2['input_ids']
]
print(batched_ids)
print("------------")

attention_mask = [
    [1]*16,
    [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
]

outputs = model(torch.tensor(batched_ids), attention_mask=torch.tensor(attention_mask))
print(outputs.logits)
print("-----------")