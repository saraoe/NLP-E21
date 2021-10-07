'''
Exercise for class 5

Exercises on using Neural networks for classification_
- Find a multilabel classification dataset on huggingface datasets
- Apply train a multiclass classification neural network on it

'''
import sys
sys.path.append("..")

import spacy
nlp = spacy.load("en_core_web_sm")
from sklearn.feature_extraction import DictVectorizer
import numpy as np
import torch
import torch.nn as nn

from class4.frequency import term_freq
from class5.neural_network_as_nnmodule import Model
from datasets import load_dataset


class Model(nn.Module):
    def __init__(self, n_input_features = 10):
        super(Model, self).__init__()
        self.linear1 = nn.Linear(n_input_features, 30)
        self.linear2 = nn.Linear(30, 30)
        self.linear3 = nn.Linear(30, 1)

    def forward(self, x):
        x = self.linear1(x)
        x = torch.softmax(x)
        x = self.linear2(x)
        x = torch.softmax(x)
        x = self.linear3(x)
        y_pred = torch.softmax(x)
        return y_pred

dataset = load_dataset("health_fact")

train = dataset['train']
# print(train.features)
# print(train['main_text'][0], train['label'][0])


tf = []
for text in train['main_text']:
    doc = nlp(text)
    tf.append(term_freq([t for t in doc]))

v = DictVectorizer(sparse=False)
X_numpy = v.fit_transform(tf)
y_numpy = np.array(train['label'])


X = torch.tensor(X_numpy, dtype=torch.float)
y = torch.tensor(y_numpy, dtype=torch.float)
y = y.view(y.shape[0], 1)


# initialize model
n_features = X.shape[1]
model = Model(n_input_features=n_features)

# define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters()) 

# train
epochs = 10000
for epoch in range(epochs):
    # forward
    y_hat = model(X)

    # backward
    loss = criterion(y_hat, y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    # some print to see that it is running
    if (epoch+1) % 1000 == 0:
        print(f'epoch: {epoch+1}, loss = {loss.item():.4f}')
