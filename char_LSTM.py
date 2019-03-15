# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 14:48:50 2019

@author: 振振
"""
#使用keras的深度学习框架
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM,GRU
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
#读入文本
raw_text = open('Winston_Churchil.txt',encoding='gb18030',errors='ignore').read()#设置编码格式，不然容易报错
raw_text = raw_text.lower()
chars = sorted(list(set(raw_text)))#有多少个字符
char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))
#数据处理，找出可训练的x和y
seq_length = 100
x=[]
y=[]
for i in range(0, len(raw_text) - seq_length):
    given = raw_text[i:i + seq_length]
    predict = raw_text[i + seq_length]
    x.append([char_to_int[char] for char in given])
    y.append(char_to_int[predict])
print(len(x))
n_patterns = len(x)
n_vocab = len(chars)

# 把x变成LSTM需要的样子
x = numpy.reshape(x, (n_patterns, seq_length, 1))
# 简单normal到0-1之间
x = x / float(n_vocab)
# output变成one-hot
y = np_utils.to_categorical(y)
#使用keras搭建模型
model = Sequential()
model.add(LSTM(256, input_shape=(x.shape[1], x.shape[2])))
#GRU是LSTM的一个变体，计算复杂度较之更低
#model.add(GRU(256, input_shape=(x.shape[1], x.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')
model.fit(x, y, nb_epoch=50, batch_size=4096，verbose=0)#batch_size设置的如果太大一般的机器带不起来，verbose参数可以设置是否显示训练过程
#下面做一个用来测试模型的函数
def predict_next(input_array):
    x = numpy.reshape(input_array, (1, seq_length, 1))
    x = x / float(n_vocab)
    y = model.predict(x)
    return y

def string_to_index(raw_input):
    res = []
    for c in raw_input[(len(raw_input)-seq_length):]:
        res.append(char_to_int[c])
    return res

def y_to_char(y):
    largest_index = y.argmax()
    c = int_to_char[largest_index]
    return c
def generate_article(init, rounds=200):
    in_string = init.lower()
    for i in range(rounds):
        n = y_to_char(predict_next(string_to_index(in_string)))
        in_string += n
    return in_string

init = 'His object in coming to New York was to engage officers for that service. He came at an opportune moment'
article = generate_article(init)
print(article)
















