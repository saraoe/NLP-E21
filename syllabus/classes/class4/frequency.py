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
        for term in set(doc):
            d[term] = d.get(term, 0) + 1
    return d


# def tf_idf(docs: list[list[str]]):
#     '''
#     term frequency-inverse document frequency
#     '''
#     docs_f = doc_freq(docs)
#     for doc in docs:
#         doc_f = term_freq(doc)
        


if __name__=='__main__':
    from sklearn.feature_extraction import DictVectorizer
    docs = [['hej', 'med', 'dig', 'hej'],
            ['jeg', 'hedder', 'bob'],
            ['hej', 'jeg', 'hedder', 'anders'],
            ['hej', 'med', 'dig', 'bob']]
    print(term_freq(docs[0]))
    print(doc_freq(docs))

    v = DictVectorizer(sparse=False)
    list_freq = [term_freq(doc) for doc in docs]
    tf = v.fit_transform(list_freq)
    print(tf)
