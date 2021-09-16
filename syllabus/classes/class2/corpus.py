'''
Corpus object for class 2 part 1
'''
import os

# Read text File
def read_text_file(file_path):
    with open(file_path, 'r', encoding="utf8") as f:
        return f.read()



class corpus:
    def __init__(self, path) -> None:
        self.ids = []
        self.texts = []
        os.chdir(path)
        # iterate through all file
        for file in os.listdir():
            if file.endswith(".txt"):
                file_path = f"{path}/{file}"
                self.ids.append(file[:-4])
                self.texts.append(read_text_file(file_path))
    
    def seg_sentences(self):
        pass

    def tokenization(self):
        pass

print(corpus('C:/Users/Sara/Documents/7semester/nlp/NLP-E21/syllabus/classes/class2/train_corpus').ids)

