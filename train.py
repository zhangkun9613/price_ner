from model import BiLSTM
from utils import batch_iter,get_data
from vocab import Vocab
from seqeval.metrics import classification_report
from torch import optim
import numpy as np
import torch

x_train,x_valid,x_test,y_train,y_valid,y_test = get_data('time_delay')
train_data = list(zip(x_train,y_train))
vocab = Vocab.from_corpus(x_train)
tag_vocab = Vocab.from_corpus(y_train)

model = BiLSTM(vocab,tag_vocab,100, 256)

optimizer = optim.Adam(model.parameters(), lr=0.01)
for epoch in range(3):
    for sents,labels in batch_iter(train_data,16):
        model.zero_grad()
        loss,acc= model(sents,labels)
        print("epoch {}:".format(epoch),loss,acc)
        loss.backward()
        optimizer.step() 

test_data = list(zip(x_test,y_test))
preds = []
for sent,labels in test_data:
    pred = model.predict([sent])
    preds.append(pred.tolist()[0])
preds = [[tag_vocab.id2word[i] for i in sent] for sent in preds]
print(classification_report(y_test,preds,digits=4))