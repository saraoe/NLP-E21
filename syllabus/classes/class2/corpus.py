'''
Corpus object for class 2
'''
import os, re

# Define functions
def read_text_file(file_path):
    with open(file_path, 'r', encoding="utf8") as f:
        return f.read()


def corpus_loader(folder: str):
    """
    A corpus loader function which takes in a path to a 
    folder.
    """
    # iterate through all file
    for file in os.listdir(folder):
        file_path = f"{folder}/{file}"
        yield read_text_file(file_path)


## Corpus class ##
class corpus:
    def __init__(self, path) -> None:
        self.ids = []
        self.texts = []
        # iterate through all file
        for file in os.listdir(path):
            if file.endswith(".txt"):
                file_path = f"{path}/{file}"
                self.ids.append(file[:-4])
                self.texts.append(read_text_file(file_path))
    
    def seg_sentences(self) -> None:
        '''
        Segments texts into sentences
        sent_list has same length as number of texts
        '''
        def f(x):
            return re.split('[.!]', x)
        self.sent_list = list(map(f, self.texts))

    def tokenization(self) -> None:
        '''
        Tokenizes the texts
        token_list has same length as number of texts
        '''
        def f(x):
            return re.split(r'\W+', x.lower())
        self.token_list = list(map(f, self.texts))
    
    def PMI(self, word: str) -> list:
        '''
        Calculate the pointwise mutual information (PMI or MI) for each word in the texts.
        
        Returns a list with same lenght as the number of texts. 
        The list contains dictionaries with the PMI for each word in each text
        If the word is not in the text, None is returned
        '''
        out = []
        for tokens in self.token_list:
            if word not in tokens:
                out.append(None)
                continue
            d = {}
            for i, token in enumerate(tokens):
                if token == word:
                    if i != 0: 
                        word_before = tokens[i-1]
                        d[word_before] = d.get(word_before, 0) + 1
                    if i != len(tokens)-1: 
                        word_after = tokens[i+1]
                        d[word_after] = d.get(word_after, 0) + 1
            out.append(d)
        return out





if __name__ == "__main__":
    path = 'C:/Users/Sara/Documents/7semester/nlp/NLP-E21/syllabus/classes/data/train_corpus'
    c = corpus(path)
    c.seg_sentences()
    c.tokenization()

    print('text = ', c.texts[0])
    # print('sent = ', c.sent_list[0])
    # print('tokens = ', c.token_list[0])
    print(c.PMI('wife')[0])

