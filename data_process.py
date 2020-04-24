import json
import os
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


root_path = "./"
def json2text():
    for name in os.listdir(root_path):
        file_name = root_path + '\\'+ name
        with open(file_name, 'r',encoding="utf-8") as f:
            data = f.readlines()
        re = []
        temp = ""
        for i in data:
            i = i.strip().split(':')
            i[0] = i[0].replace('"','').strip()
            if(i[0] == 'evidence' and i[1].find('<')<0 and i[1].find('>')<0):
                re.append(i[1].replace('"','').strip())
        print(len(re),len(data),name)
        with open(name.replace('json','txt'),'w',encoding="utf-8") as f:
            f.write("\n".join(re))

def parse_jsonline(file_name):
    with open(file_name, 'r',encoding="utf-8") as f:
        lines = f.readlines()
    lines = [eval(i.strip().replace('null','None')) for i in lines]
    return lines

def json_to_train_data(name):
    file_name = "./data/raw_data/" + "{}.json1".format(name)
    data = parse_jsonline(file_name)
    count = 0
    re = []
    tag_names = {'金额':'price','不存在':''}
    for sample in data:
        text = sample['text']
        labels_info = sample['labels']
        if labels_info == [] or text.find('>') >= 0 or text.find('<') >= 0:
            continue
        labels = ['O' for i in range(len(text))]
        text = list(text.strip())
        for tag in labels_info:
            tag_name = tag_names[tag[2]]
            if(tag_name == ''):
                continue
            for pos in range(tag[0], tag[1], 1):
                labels[pos] = 'I-{}'.format(tag_name)
        text = list(zip(text, labels))
        #将空格换成‘，’
        for i in range(len(text)-1, -1, -1):
            if text[i][0] == ' ':
                if i == 0 or text[i-1][0] == ' ':
                    del text[i]
                else:
                    text[i] = (',', 'O')
        #添加 B
        for i in range(len(text)):
            if text[i][1][0] == 'I':
                if i==0 or text[i-1][1][0] == 'O':
                    text[i] = (text[i][0],'B'+text[i][1][1:])
        text = ['\t'.join(i) for i in text]
        re.append('\n'.join(text)+'\n')
        count += 1
    #re[-1] = re[-1].strip()
    with open('./data/train_data/{}.train'.format(name), 'w', encoding='utf-8') as f:
        f.write('\n'.join(re))
    print(count)

if __name__ == '__main__':
    file_names = ['ask_price','time_delay','price']
    for name in file_names:
        json_to_train_data(name)
