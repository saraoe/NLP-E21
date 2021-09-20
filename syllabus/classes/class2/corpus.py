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
        self.sent_list= list(map(lambda x: re.split('[.!]', x), self.texts))

    def tokenization(self) -> None:
        '''
        Tokenizes the texts
        token_list has same length as number of texts
        '''
        self.token_list = list(map(lambda x: re.split(r'\W+', x), self.texts))
    
    def PMI(self) -> list:
        '''
        Calculate the pointwise mutual information (PMI or MI) for each word in the texts.
        
        Returns a list with same lenght as the number of texts. 
        The list contains dictionaries with the PMI for each word in each text
        '''
        pass


if __name__ == "__main__":
    path = 'C:/Users/Sara/Documents/7semester/nlp/NLP-E21/syllabus/classes/data/train_corpus'
    c = corpus(path)
    c.seg_sentences()
    c.tokenization()

    print('text = ', c.texts[0])
    print('sent = ', c.sent_list[0])
    print('tokens = ', c.token_list[0])

