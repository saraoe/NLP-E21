'''
Exercises for class 3
'''
import os
import spacy
from collections import Counter
nlp = spacy.load("en_core_web_sm")

# corpus loader
def corpus_loader(folder: str):
    """
    A corpus loader function which takes in a path to a 
    folder.
    """
    # iterate through all file
    for file in os.listdir(folder):
        file_path = f"{folder}/{file}"
        with open(file_path, 'r', encoding="utf8") as f:
            yield f.read()


## exercises ##

# load txt-files
path = 'C:/Users/Sara/Documents/7semester/nlp/NLP-E21/syllabus/classes/data/train_corpus'
texts = list(corpus_loader(path))

# 2) Filter a text to keep only the lemma of nouns, adjectives and verbs
doc = nlp(texts[0])
clean = []
for token in doc:
    if token.pos_ in ['NOUN','ADJ','VERB']:
        clean.append(token.lemma_)
# print(clean)

# 3) Calculate the ratio of pos-tags in texts.
pos_tags = Counter([token.pos_ for token in doc])
n = len(doc)
pos_tags_ratio = [(tag, count/n) for tag, count in pos_tags.items()]
# print(pos_tags)
print(pos_tags_ratio)

# 4) Calculate the mean dependency distance (MDD) for one of the texts.
mdd = sum([abs(token.i-token.head.i) for token in doc])/n
print(mdd)
