'''
Preparation to class 4

Functions for class:
- Term frequency
- Document frequency
'''
from collections import Counter


def term_freq(tokens: list[str]) -> dict:
    """
    Takes in a list of tokens (str) and return a dictionary of term frequency of each token
    """
    term_count = Counter(tokens)
    n = len(tokens)
    return {term: count/n for term, count in term_count.items()}


def doc_freq(docs: list[list[str]]) -> dict:
    """
    Takes in a list of documents which each is a list of tokens and return a dictionary of
    frequencies for each token over all the documents. 
    
    E.g. {"Aarhus": 20, "the": 2301, ...}
    """
    d = {}
    for doc in docs:
        term_count = Counter(doc)
        for term, count in term_count.items():
            d[term] = d.get(term, 0) + count
    return d


if __name__=='__main__':
    docs = [['hej', 'med', 'dig', 'hej'],
            ['jeg', 'hedder', 'bob'],
            ['hej', 'jeg', 'hedder', 'anders'],
            ['hej', 'med', 'dig', 'bob']]
    print(term_freq(docs[0]))
    print(doc_freq(docs))