"""
Different examples of using pipeline module
"""
from transformers import pipeline

# Sentiment analysis
classifier = pipeline("sentiment-analysis")
classifier_output = classifier([
    "I am soo sad",
    "The weather is beautiful"
])
print(classifier_output)
print("----------")

# Zero shot classification
classifier = pipeline("zero-shot-classification")
classifier_output = classifier(
    "This is a course about the Transformers library",
    candidate_labels=["education", "politics", "business"],
)
print(classifier_output)
print("----------")

# Text generation
generator = pipeline("text-generation")
print(generator("In this course, we will teach you how to", num_return_sequences=2, max_length=30))
print("--------")

# Translation
# Gave some confusing sentences in Telugu
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-mul-en")
print(translator([
    "మీరు ఎలా ఉన్నారు",
    "ఇంకా వెళ్లి వస్తాను",
    "ఆమె ఒక భార్య", 
    "Nuvvu evaru"
]))
print("----------")

#Mask filling - BERT objective
# <mask> can be [MASK] depending on which model is passed to pipeline
unmasker = pipeline("fill-mask")
print(unmasker("This course will teach you all about <mask> models.", top_k=2))
print("-------------")


# Named Entity Recognition
# Inside the pipeline, each token is split into subtokens and again at the end are grouped
# as we gave grouped_entities = True
ner = pipeline("ner", grouped_entities=True)
print(ner("My name is Sylvain and I work at Hugging Face in Brooklyn."))
print("-------------")
# Question answering
# Generating an answer to a question from the given context
question_answerer = pipeline("question-answering")
print(question_answerer(
    question="Where do I work?",
    context="My name is Sylvain and I work at Hugging Face in Brooklyn",
))
print("-------------")





