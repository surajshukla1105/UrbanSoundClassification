# -*- coding: utf-8 -*-
"""
Created on Sun Mar 25 01:12:47 2018

@author: Suraj Shukla
"""

import librosa
data, sampling_rate = librosa.load('D:/1 DataScience Code and Data/1. AV/Urban Sound Classification/train/Train/2022.wav')

import os
import pandas as pd
import librosa
import librosa.display


root_dir = os.path.abspath('.')
data_dir = '../Urban Sound Classification'

train = pd.read_csv(os.path.join(data_dir, 'train/train.csv'))
test = pd.read_csv(os.path.join(data_dir, 'test/test.csv'))

# checking distribution 
train.Class.value_counts()

# Reading data
def parser(row):
   # function to load files and extract features
   file_name = os.path.join(os.path.abspath(data_dir), 'train\\Train', str(row.ID) + '.wav')

   # handle exception to check if there isn't a file which is corrupted
   try:
      # here kaiser_fast is a technique used for faster extraction
      X, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 
      # we extract mfcc feature from data
      mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0) 
   except Exception as e:
      print("Error encountered while parsing file: ", file_name)
      return None, None
 
   feature = mfccs
   label = row.Class
 
   return [feature, label]

temp = train.apply(parser, axis=1)
temp.columns = ['feature', 'label']
 

#Convert the data to pass it in our deep learning model
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import Adam
from keras.utils import np_utils
from sklearn import metrics 
from sklearn.preprocessing import LabelEncoder
X = np.array(temp.feature.tolist())
y = np.array(temp.label.tolist())

lb = LabelEncoder()

y = np_utils.to_categorical(lb.fit_transform(y))

#Step 4: Run a deep learning model and get results


num_labels = y.shape[1]
filter_size = 4

# build model
model = Sequential()

model.add(Dense(256, input_shape=(40,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.5))


model.add(Dense(num_labels))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')


model.fit(X, y, batch_size=32, epochs=80,validation_split=0.2)

