'''
Preparation to class 4

Functions for A2:
- Term frequency
- Inverse document frequency

Based on the definition of tf-idf in slides from lecture 4
'''
from collections import Counter
from math import log10



def tf(tokens: list[str]) -> dict:
    """
    Takes in a list of tokens (str) and return a dictionary of term frequency of each token
    """
    term_count = Counter(tokens)
    return {term: log10(count+1) for term, count in term_count.items()}


def idf(docs: list[list[str]]) -> dict:
    """
    Takes in a list of documents which each is a list of tokens and return a dictionary of
    frequencies for each token over all the documents. 
    
    E.g. {"Aarhus": 20, "the": 2301, ...}
    """
    n = len(docs)
    d = {}
    for doc in docs:
        for term in set(doc):
            df = d.get(term, 0) + 1
            d[term] = log10(n / df)
    return d


def tfidf(texts: list, df: dict=None) -> list[dict]:
    """
    takes in a list of tokenized texts and returns a list of dictionaries

    args:
        df (dict): Document frequencies, defaults to None, in which case it is estimated from the texts.
    """
    if not df:
        df = idf(texts)
    
    out = []
    for text in texts:
        tf_ = tf(text)
        tmp = {term: freq*df[term] for term, freq in tf_.items()} 
        out.append(tmp)
    return out 


if __name__=='__main__':
    from sklearn.feature_extraction import DictVectorizer
    docs = [['hej', 'med', 'dig', 'hej'],
            ['jeg', 'hedder', 'bob'],
            ['hej', 'jeg', 'hedder', 'anders'],
            ['hej', 'med', 'dig', 'bob']]


    v = DictVectorizer(sparse=False)
    list_freq = tfidf(docs)
    # list_freq = [tf(doc) for doc in docs]
    # tf = v.fit_transform(list_freq)
    tf = v.fit_transform(list_freq)
    print(tf)
