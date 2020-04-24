import math
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_data(file_name,p = 0.6,add_intent = False):
    with open("./data/train_data/{}.train".format(file_name),'r') as f:
        lines = f.readlines()

    train_x = []
    train_y = []
    x = []; y = []
    for line in lines:
        line = line.strip().split('\t')
        if(len(line)>1):
            x.append(line[0])
            y.append(line[1])
        else:
            if add_intent:
                x.append('intent');y.append('B-'+file_name+'_intent')
            train_x.append(x);train_y.append(y)
            x = [];y = []
    if(x!=[]):
        if add_intent:
            x.append('intent');y.append('B-'+file_name+'_intent')
        train_x.append(x);train_y.append(y)
    nums = len(train_x)
    x_train = train_x[0:int(p*nums)];x_valid = train_x[int(0.8*nums):int(0.9*nums)];x_test = train_x[int(0.9*nums):]
    y_train = train_y[0:int(p*nums)];y_valid = train_y[int(0.8*nums):int(0.9*nums)];y_test = train_y[int(0.9*nums):]
    return x_train,x_valid,x_test,y_train,y_valid,y_test

def get_combined_data(p=0.6,add_intent = False):
    x_train = [];x_valid=[];x_test=[];y_train=[];y_valid=[];y_test=[]
    for file_name in ['ask_price','time_delay']:
        x_train_t,x_valid_t,x_test_t,y_train_t,y_valid_t,y_test_t = get_data(file_name,p,add_intent)
        x_train.extend(x_train_t);x_valid.extend(x_valid_t);x_test.extend(x_test_t)
        y_train.extend(y_train_t);y_valid.extend(y_valid_t);y_test.extend(y_test_t)
    return x_train,x_valid,x_test,y_train,y_valid,y_test

def get_combined_data_sep_label(p=0.6):
    x_train = [];x_valid=[];x_test=[];y_train=[];y_valid=[];y_test=[]
    for file_name in ['ask_price','time_delay','price']:
        x_train_t,x_valid_t,x_test_t,y_train_t,y_valid_t,y_test_t = get_data(file_name,p)

        y_train_t = [[i.replace('price','price.{}'.format(file_name)) for i in sample] for sample in y_train_t]
        y_valid_t = [[i.replace('price','price.{}'.format(file_name)) for i in sample] for sample in y_valid_t]
        y_test_t = [[i.replace('price','price.{}'.format(file_name)) for i in sample] for sample in y_test_t]
        
        x_train.extend(x_train_t);x_valid.extend(x_valid_t);x_test.extend(x_test_t)
        y_train.extend(y_train_t);y_valid.extend(y_valid_t);y_test.extend(y_test_t)
    return x_train,x_valid,x_test,y_train,y_valid,y_test

def batch_iter(data, batch_size, shuffle=False):
    batch_num = math.ceil(len(data) / batch_size)
    index_array = list(range(len(data)))

    if shuffle:
        np.random.shuffle(index_array)

    for i in range(batch_num):
        indices = index_array[i * batch_size: (i + 1) * batch_size]
        examples = [data[idx] for idx in indices]

        examples = sorted(examples, key=lambda e: len(e[0]), reverse=True)
        src_sents = [e[0] for e in examples]
        tgt_sents = [e[1] for e in examples]

        yield src_sents, tgt_sents

