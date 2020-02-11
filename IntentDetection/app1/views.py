# -*- coding: utf-8 -*-

# Create your views here.
from __future__ import unicode_literals

from django.shortcuts import render
from django.http import HttpResponse
from .forms import QuestionForm 
from . import templates
from django import template
from keras.preprocessing.text import Tokenizer
from keras.optimizers import SGD, Adam, Nadam, RMSprop
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.preprocessing import sequence
from keras.models import Sequential,Model,load_model
from keras.layers import Embedding,Conv1D,MaxPooling1D, SpatialDropout1D
from keras.layers.core import Dense, Activation,Dropout ,Flatten
from keras.layers.recurrent import LSTM
from keras.utils import np_utils
from keras.constraints import maxnorm
from keras.callbacks import ModelCheckpoint,TensorBoard, ReduceLROnPlateau,EarlyStopping
from keras.applications import Xception
from keras import regularizers
from keras import backend as K
from keras.preprocessing import sequence
from sklearn.utils import shuffle

# Create your views here.

import pandas as pd
import numpy as np
import json
import os
import random

seed = 120
np.random.seed(seed)

data_list = []
max_words = 200
def create_data(flag , data):
    for each in data:
        sent = ''
        for item in each['data']:
            sent += item['text']
        l = [sent, 0, 0, 0, 0]
        l[flag] = 1
        data_list.append(l)
        

path = '/home/kolaparthi/Downloads/data/'
train_files = [file for file in os.listdir(path) if file.endswith('.json')]
flag = 1
for each in train_files:
    with open(path+each) as file:
        df = json.load(file)
        
    s_index = 6
    e_index = each.find('_full')
    name = each[s_index:e_index] 
    
    json_data = df[name]
    create_data(flag, json_data)
    flag += 1
    print(name)

data_frame = pd.DataFrame(data_list, columns=['sentence', 'SearchScreeningEvent', 'SearchCreativeWork', 'RateBook', 'AddToPlaylist'])
data_frame.to_csv(path+'train.csv')

csv_data = pd.read_csv(path+'train.csv')
csv_data = csv_data.drop('Unnamed: 0', axis = 1)

df = shuffle(csv_data)
df.head()

X_train = df['sentence'].values
Y_train = df[['SearchScreeningEvent', 'SearchCreativeWork', 'RateBook', 'AddToPlaylist']].values


def preparingDataForTraining(X_train, Y_train):
   
    labels = np.array([1,1,1,1,1,0,0,0,0,0])
    # prepare tokenizer
    t = Tokenizer(oov_token=True)
    t.fit_on_texts(X_train)
    vocab_size = len(t.word_index) + 1
    # integer encode the documents
    encoded_docs = t.texts_to_sequences(X_train)


    val_samples = 1000 # Test samples for validation
    x_train = X_train[val_samples:] 
    y_train =Y_train[val_samples:] 
    x_val = X_train[:val_samples] 
    y_val = Y_train[:val_samples] 
    X_train_numbers = t.texts_to_sequences(x_train)
    X_val_numbers = t.texts_to_sequences(x_val)
    X_train_padded = sequence.pad_sequences(X_train_numbers, maxlen=max_words)
    X_valid_padded = sequence.pad_sequences(X_val_numbers, maxlen=max_words)
    return (t, X_train_padded, X_valid_padded, y_train , y_val)



def build_model():
    model = Sequential()

    model.add(Embedding(input_dim=7166, output_dim=32, input_length=max_words)) #to change words to ints
# model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
# model.add(MaxPooling1D(pool_size=2))
# model.add(Dropout(0.5))
# model.add(Conv1D(filters=32, kernel_size=2, padding='same', activation='relu'))
# model.add(MaxPooling1D(pool_size=2))
 #hidden layers
    model.add(LSTM(10))
# model.add(Flatten())
    model.add(Dropout(0.6))
    model.add(Dense(1200, activation='relu',W_constraint=maxnorm(1)))
# model.add(Dropout(0.6))
    model.add(Dense(500, activation='relu',W_constraint=maxnorm(1)))

# model.add(Dropout(0.5))
 #output layer
    model.add(Dense(4, activation='softmax'))
    return model


def trainAndFit(X_train_padded, X_valid_padded, y_train , y_val):
    model = build_model()
    learning_rate=0.0001
    epochs = 2
    batch_size = 32 #32
    sgd = SGD(lr=learning_rate, nesterov=True, momentum=0.7, decay=1e-4)
    Nadam1 = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
    model.compile(loss='categorical_crossentropy', optimizer=Nadam1, metrics=['accuracy'])
    model.predict(X_valid_padded)

    history  = model.fit(X_train_padded,y_train, epochs = epochs, batch_size=batch_size, verbose=1,
                    validation_data=(X_valid_padded, y_val))
    
    return model




def predict_helper(question):
    
    tokenizer, X_train_padded, X_valid_padded, y_train , y_val = preparingDataForTraining(X_train, Y_train)
    model = trainAndFit(X_train_padded, X_valid_padded, y_train , y_val)
    
    g = tokenizer.texts_to_sequences([question])
    print(question)
    print("The value of g is ", g)
    s = sequence.pad_sequences(g, maxlen=max_words)
    pred = model.predict(np.array(s))
    i,j = np.where(pred== pred.max()) #calculates the index of the maximum element of the array across all axis
    if(pred.max() <= 0.2):
        print("It doesn't belong to any of this category")
    else:
        i = int(i)
        j = int(j)
        print(pred)
        total_possible_outcomes = ['SearchSreeningEvent','SearchCreativeWork','RateBook','AddToPlaylist']
        return total_possible_outcomes[j]


def predict(request):
    if request.method == 'POST':
        form = QuestionForm(request.POST)
        if form.is_valid():
            intent = predict_helper(request.POST['test_statement'])
            return HttpResponse(request.POST['test_statement']+" belongs to the intent "+intent)
    else:
        form = QuestionForm()
    return render(request, 'predict.html', {'form':form})
