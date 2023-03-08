#!/usr/bin/env python
# coding: utf-8

# In[2]:


from scipy import signal
import matplotlib as mpl


#Importing necessary modules
import numpy as np
import math
import scipy.signal as signal
import pylab as pl
import matplotlib.pyplot as plt
import pandas as pd

#Plotting FHR Dataset Row 0
import matplotlib.pyplot as plt


from scipy import interpolate
file = open('/Users/sarah/Downloads/RawFHR.csv')
df2 = np.loadtxt(file,delimiter=",")
rawdata = np.array(df2)

rawdata = rawdata.transpose()
print(rawdata)


# Interpolation
NewRecordings = []
for row in rawdata:
    recording = row
    recording = recording[np.logical_not(np.isnan(recording))]   # for removing nan
    #print(recording)
    L = len(recording)
    for i in range(L):
        if recording[i]==0:
            recording[i]=float("Nan")
    axisX = np.array(range(L))
    Y = np.array(recording)
    idx_finite=np.isfinite(Y)
    f_finite = interpolate.interp1d(axisX[idx_finite], Y[idx_finite], kind = 'cubic', bounds_error=False)
    ynew_finite = f_finite(axisX)
    #print(ynew_finite)
    NewRecordings.append(ynew_finite)
   


# In[3]:



# For making data to be same size
row_lengths = []
ProRecordings = []
for row in NewRecordings:
    row_lengths.append(len(row))
max_length = max(row_lengths)
print(max_length)
for row in NewRecordings:
    row = row.tolist()
    while len(row) < max_length:
        row.append(float("nan"))
    ProRecordings.append(row)
   
ProRecordings = np.array(ProRecordings)    
print(ProRecordings)

# Segmentation

#Step1: remove nans
edit = []
for row in ProRecordings:
    new = row[np.logical_not(np.isnan(row))]
    #print(new)
    #new = new.tolist()
    #print(new.tolist())
    edit.append(new)
#edit = np.array(edit)
#edit = edit.tolist()
print(edit)


# In[4]:


# Step2: check the length of recordings
lengths = []
for array in edit:
    l = (len(array))
    lengths.append(l)
print(lengths)

# Step3: segmentation
#segment_length = 1   # unit: min
                     # convert length to samples
                     # initial empty list
       
  # In this step, we will have a matrix (10*240)

#time = []
#for val in lengths:
    #minutes = val//60//4
    #time.append(minutes)
#print(time)


# In[5]:


biggerseg = 10
min10segment = []
for array in edit:
    sample = array
    segment_10min = sample[biggerseg*60*4*-1-1:-1]
    min10segment.append(segment_10min)
tenminseg = np.array(min10segment)
print(tenminseg)
print(tenminseg.shape)

smallerseg = 1
min1segment = []
step = smallerseg*60*4
NumofSeg = biggerseg//smallerseg
SmallDataSegment = []
for row in tenminseg:
    pointer = 0
    for idx in range(NumofSeg):
        if idx==1:
            smallsegment = row[pointer-step:]
        else:
            smallsegment = row[pointer-step:pointer]
        SmallDataSegment.append(smallsegment)
        pointer = pointer-step

       
#SmallDataSegment = np.array(SmallDataSegment)
#print(SmallDataSegment.shape)
#for row in SmallDataSegment:
    #print(np.array(row))
smallarray = np.array(SmallDataSegment)
print(smallarray.shape)


# In[6]:




#Step 4: Label the Data
import csv
import numpy as np

#threshold = 7.15, meaning if it was above 7.15 it was normal and labeled as 1

label = []

file_1 = open('/Users/sarah/Downloads/ph Values.csv')
clinical_info = np.loadtxt(file_1, delimiter=',', skiprows = 1)
print(len(clinical_info))
for val in clinical_info:
    NumofSeg = biggerseg//smallerseg
    while NumofSeg>=1:
        if val >= 7.15:
            new_column = 1
        else:
            new_column = 0
        NumofSeg = NumofSeg - 1
        label.append(new_column)
        #label.append([val,new_column])
       
#print(labeled)
labeled = np.array(label)
       
print(labeled)
print(len(labeled))
#np.savetxt("labeleddata.csv", label, delimiter = ',')


# In[12]:


# Step6: convert segments to images

import pywt
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import glob
from os import path

outpath = 'C:/Users/rmnin/Downloads/cwt images'
indices =[]
#for i in range(len(SmallDataSegment)):
import time

start = time.time()

X = []


    
for i in range(927,1000):    
    row = SmallDataSegment[i]
    cwtmatr, freqs = pywt.cwt(row, np.arange(1, 552, 10), 'morl', method = 'fft' )
        #print(cwtmatr.shape)

    img = plt.imshow(abs(cwtmatr), extent = [0, 20000, 1, 20000], cmap='gray', aspect='auto')
        #i += 1
    plt.savefig('/Users/sarah/Downloads/cwt images/' + str(i) + '.png')
    
for i in range(999,1250):    
    row = SmallDataSegment[i]
    cwtmatr, freqs = pywt.cwt(row, np.arange(1, 552, 10), 'morl', method = 'fft' )
        #print(cwtmatr.shape)

    img = plt.imshow(abs(cwtmatr), extent = [0, 20000, 1, 20000], cmap='gray', aspect='auto')
        #i += 1
    plt.savefig('/Users/sarah/Downloads/cwt images/' + str(i) + '.png')
    
for i in range(1249,1500):    
    row = SmallDataSegment[i]
    cwtmatr, freqs = pywt.cwt(row, np.arange(1, 552, 10), 'morl', method = 'fft' )
        #print(cwtmatr.shape)

    img = plt.imshow(abs(cwtmatr), extent = [0, 20000, 1, 20000], cmap='gray', aspect='auto')
        #i += 1
    plt.savefig('/Users/sarah/Downloads/cwt images/' + str(i) + '.png')
for i in range(1499,1750):    
    row = SmallDataSegment[i]
    cwtmatr, freqs = pywt.cwt(row, np.arange(1, 552, 10), 'morl', method = 'fft' )
        #print(cwtmatr.shape)

    img = plt.imshow(abs(cwtmatr), extent = [0, 20000, 1, 20000], cmap='gray', aspect='auto')
        #i += 1
    plt.savefig('/Users/sarah/Downloads/cwt images/' + str(i) + '.png')

for i in range(1749,2000):    
    row = SmallDataSegment[i]
    cwtmatr, freqs = pywt.cwt(row, np.arange(1, 552, 10), 'morl', method = 'fft' )
        #print(cwtmatr.shape)

    img = plt.imshow(abs(cwtmatr), extent = [0, 20000, 1, 20000], cmap='gray', aspect='auto')
        #i += 1
    plt.savefig('/Users/sarah/Downloads/cwt images/' + str(i) + '.png')
for i in range(1999,2250):    
    row = SmallDataSegment[i]
    cwtmatr, freqs = pywt.cwt(row, np.arange(1, 552, 10), 'morl', method = 'fft' )
        #print(cwtmatr.shape)

    img = plt.imshow(abs(cwtmatr), extent = [0, 20000, 1, 20000], cmap='gray', aspect='auto')
        #i += 1
    plt.savefig('/Users/sarah/Downloads/cwt images/' + str(i) + '.png')
for i in range(2249,2500):    
    row = SmallDataSegment[i]
    cwtmatr, freqs = pywt.cwt(row, np.arange(1, 552, 10), 'morl', method = 'fft' )
        #print(cwtmatr.shape)

    img = plt.imshow(abs(cwtmatr), extent = [0, 20000, 1, 20000], cmap='gray', aspect='auto')
        #i += 1
    plt.savefig('/Users/sarah/Downloads/cwt images/' + str(i) + '.png')
for i in range(2499,2750):    
    row = SmallDataSegment[i]
    cwtmatr, freqs = pywt.cwt(row, np.arange(1, 552, 10), 'morl', method = 'fft' )
        #print(cwtmatr.shape)

    img = plt.imshow(abs(cwtmatr), extent = [0, 20000, 1, 20000], cmap='gray', aspect='auto')
        #i += 1
    plt.savefig('/Users/sarah/Downloads/cwt images/' + str(i) + '.png')
for i in range(2749,3000):    
    row = SmallDataSegment[i]
    cwtmatr, freqs = pywt.cwt(row, np.arange(1, 552, 10), 'morl', method = 'fft' )
        #print(cwtmatr.shape)

    img = plt.imshow(abs(cwtmatr), extent = [0, 20000, 1, 20000], cmap='gray', aspect='auto')
        #i += 1
    plt.savefig('/Users/sarah/Downloads/cwt images/' + str(i) + '.png')
for i in range(2999,3250):    
    row = SmallDataSegment[i]
    cwtmatr, freqs = pywt.cwt(row, np.arange(1, 552, 10), 'morl', method = 'fft' )
        #print(cwtmatr.shape)

    img = plt.imshow(abs(cwtmatr), extent = [0, 20000, 1, 20000], cmap='gray', aspect='auto')
        #i += 1
    plt.savefig('/Users/sarah/Downloads/cwt images/' + str(i) + '.png')
for i in range(3249,3500):    
    row = SmallDataSegment[i]
    cwtmatr, freqs = pywt.cwt(row, np.arange(1, 552, 10), 'morl', method = 'fft' )
        #print(cwtmatr.shape)

    img = plt.imshow(abs(cwtmatr), extent = [0, 20000, 1, 20000], cmap='gray', aspect='auto')
        #i += 1
    plt.savefig('/Users/sarah/Downloads/cwt images/' + str(i) + '.png')
for i in range(3499,3750):    
    row = SmallDataSegment[i]
    cwtmatr, freqs = pywt.cwt(row, np.arange(1, 552, 10), 'morl', method = 'fft' )
        #print(cwtmatr.shape)

    img = plt.imshow(abs(cwtmatr), extent = [0, 20000, 1, 20000], cmap='gray', aspect='auto')
        #i += 1
    plt.savefig('/Users/sarah/Downloads/cwt images/' + str(i) + '.png')
for i in range(3749,4000):    
    row = SmallDataSegment[i]
    cwtmatr, freqs = pywt.cwt(row, np.arange(1, 552, 10), 'morl', method = 'fft' )
        #print(cwtmatr.shape)

    img = plt.imshow(abs(cwtmatr), extent = [0, 20000, 1, 20000], cmap='gray', aspect='auto')
        #i += 1
    plt.savefig('/Users/sarah/Downloads/cwt images/' + str(i) + '.png')
    
for i in range(3999,4250):    
    row = SmallDataSegment[i]
    cwtmatr, freqs = pywt.cwt(row, np.arange(1, 552, 10), 'morl', method = 'fft' )
        #print(cwtmatr.shape)

    img = plt.imshow(abs(cwtmatr), extent = [0, 20000, 1, 20000], cmap='gray', aspect='auto')
        #i += 1
    plt.savefig('/Users/sarah/Downloads/cwt images/' + str(i) + '.png')
for i in range(4249,4500):    
    row = SmallDataSegment[i]
    cwtmatr, freqs = pywt.cwt(row, np.arange(1, 552, 10), 'morl', method = 'fft' )
        #print(cwtmatr.shape)

    img = plt.imshow(abs(cwtmatr), extent = [0, 20000, 1, 20000], cmap='gray', aspect='auto')
        #i += 1
    plt.savefig('/Users/sarah/Downloads/cwt images/' + str(i) + '.png')
    
for i in range(4499,4750):    
    row = SmallDataSegment[i]
    cwtmatr, freqs = pywt.cwt(row, np.arange(1, 552, 10), 'morl', method = 'fft' )
        #print(cwtmatr.shape)

    img = plt.imshow(abs(cwtmatr), extent = [0, 20000, 1, 20000], cmap='gray', aspect='auto')
        #i += 1
    plt.savefig('/Users/sarah/Downloads/cwt images/' + str(i) + '.png')
for i in range(4749,5000):    
    row = SmallDataSegment[i]
    cwtmatr, freqs = pywt.cwt(row, np.arange(1, 552, 10), 'morl', method = 'fft' )
        #print(cwtmatr.shape)

    img = plt.imshow(abs(cwtmatr), extent = [0, 20000, 1, 20000], cmap='gray', aspect='auto')
        #i += 1
    plt.savefig('/Users/sarah/Downloads/cwt images/' + str(i) + '.png')
for i in range(4999,5000):    
    row = SmallDataSegment[i]
    cwtmatr, freqs = pywt.cwt(row, np.arange(1, 552, 10), 'morl', method = 'fft' )
        #print(cwtmatr.shape)

    img = plt.imshow(abs(cwtmatr), extent = [0, 20000, 1, 20000], cmap='gray', aspect='auto')
        #i += 1
    plt.savefig('/Users/sarah/Downloads/cwt images/' + str(i) + '.png')
        

array= np.array(X)              
print(np.array(X))

end = time.time()
print(end-start) 


# In[6]:


import pywt
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import glob
from os import path
Y = []
for i in range(0,500):    
    #row = SmallDataSegment[i]
    #if np.prod(np.array(row)) == 0:
        #indices.append(i)
   
    #else:
    cwtmatr, freqs = pywt.cwt(row, np.arange(1, 552, 10), 'morl', method = 'fft' )
        #print(cwtmatr.shape)
    Y.append(np.array(cwtmatr))
    #img = plt.imshow(abs(cwtmatr), extent = [0, 20000, 1, 20000], cmap='gray', aspect='auto')
        #i += 1
        #plt.savefig('C:/Users/rmnin/Downloads/cwt images/' + str(i) + '.png')
for i in range(500,1000):    
    #row = SmallDataSegment[i]
    #if np.prod(np.array(row)) == 0:
        #indices.append(i)
   
    #else:
    cwtmatr, freqs = pywt.cwt(row, np.arange(1, 552, 10), 'morl', method = 'fft' )
        #print(cwtmatr.shape)
    Y.append(np.array(cwtmatr))
    #img = plt.imshow(abs(cwtmatr), extent = [0, 20000, 1, 20000], cmap='gray', aspect='auto')
        #i += 1
        #plt.savefig('C:/Users/rmnin/Downloads/cwt images/' + str(i) + '.png')
       
for i in range(1000,1500):    
    #row = SmallDataSegment[i]
    #if np.prod(np.array(row)) == 0:
        #indices.append(i)
   
    #else:
    cwtmatr, freqs = pywt.cwt(row, np.arange(1, 552, 10), 'morl', method = 'fft' )
        #print(cwtmatr.shape)
    Y.append(np.array(cwtmatr))
    #img = plt.imshow(abs(cwtmatr), extent = [0, 20000, 1, 20000], cmap='gray', aspect='auto')
        #i += 1
        #plt.savefig('C:/Users/rmnin/Downloads/cwt images/' + str(i) + '.png')
for i in range(1500,2000):    
    #row = SmallDataSegment[i]
    #if np.prod(np.array(row)) == 0:
        #indices.append(i)
   
    #else:
    cwtmatr, freqs = pywt.cwt(row, np.arange(1, 552, 10), 'morl', method = 'fft' )
        #print(cwtmatr.shape)
    Y.append(np.array(cwtmatr))
    #img = plt.imshow(abs(cwtmatr), extent = [0, 20000, 1, 20000], cmap='gray', aspect='auto')
        #i += 1
        #plt.savefig('C:/Users/rmnin/Downloads/cwt images/' + str(i) + '.png')

for i in range(2000,2500):    
    #row = SmallDataSegment[i]
    #if np.prod(np.array(row)) == 0:
        #indices.append(i)
   
    #else:
    cwtmatr, freqs = pywt.cwt(row, np.arange(1, 552, 10), 'morl', method = 'fft' )
        #print(cwtmatr.shape)
    Y.append(np.array(cwtmatr))
    #img = plt.imshow(abs(cwtmatr), extent = [0, 20000, 1, 20000], cmap='gray', aspect='auto')
        #i += 1
        #plt.savefig('C:/Users/rmnin/Downloads/cwt images/' + str(i) + '.png')
for i in range(2500,3000):    
    #row = SmallDataSegment[i]
    #if np.prod(np.array(row)) == 0:
        #indices.append(i)
   
    #else:
    cwtmatr, freqs = pywt.cwt(row, np.arange(1, 552, 10), 'morl', method = 'fft' )
        #print(cwtmatr.shape)
    Y.append(np.array(cwtmatr))
    #img = plt.imshow(abs(cwtmatr), extent = [0, 20000, 1, 20000], cmap='gray', aspect='auto')
        #i += 1
        #plt.savefig('C:/Users/rmnin/Downloads/cwt images/' + str(i) + '.png')

for i in range(3000,3500):    
    #row = SmallDataSegment[i]
    #if np.prod(np.array(row)) == 0:
        #indices.append(i)
   
    #else:
    cwtmatr, freqs = pywt.cwt(row, np.arange(1, 552, 10), 'morl', method = 'fft' )
        #print(cwtmatr.shape)
    Y.append(np.array(cwtmatr))
    #img = plt.imshow(abs(cwtmatr), extent = [0, 20000, 1, 20000], cmap='gray', aspect='auto')
        #i += 1
        #plt.savefig('C:/Users/rmnin/Downloads/cwt images/' + str(i) + '.png')

for i in range(3500,4000):    
    #row = SmallDataSegment[i]
    #if np.prod(np.array(row)) == 0:
        #indices.append(i)
   
    #else:
    cwtmatr, freqs = pywt.cwt(row, np.arange(1, 552, 10), 'morl', method = 'fft' )
        #print(cwtmatr.shape)
    Y.append(np.array(cwtmatr))
    #img = plt.imshow(abs(cwtmatr), extent = [0, 20000, 1, 20000], cmap='gray', aspect='auto')
        #i += 1
        #plt.savefig('C:/Users/rmnin/Downloads/cwt images/' + str(i) + '.png')

for i in range(4000,4500):    
    #row = SmallDataSegment[i]
    #if np.prod(np.array(row)) == 0:
        #indices.append(i)
   
    #else:
    cwtmatr, freqs = pywt.cwt(row, np.arange(1, 552, 10), 'morl', method = 'fft' )
        #print(cwtmatr.shape)
    Y.append(np.array(cwtmatr))
    #img = plt.imshow(abs(cwtmatr), extent = [0, 20000, 1, 20000], cmap='gray', aspect='auto')
        #i += 1
        #plt.savefig('C:/Users/rmnin/Downloads/cwt images/' + str(i) + '.png')
for i in range(4500,5000):    
    #row = SmallDataSegment[i]
    #if np.prod(np.array(row)) == 0:
        #indices.append(i)
   
    #else:
    cwtmatr, freqs = pywt.cwt(row, np.arange(1, 552, 10), 'morl', method = 'fft' )
        #print(cwtmatr.shape)
    Y.append(np.array(cwtmatr))
    #img = plt.imshow(abs(cwtmatr), extent = [0, 20000, 1, 20000], cmap='gray', aspect='auto')
        #i += 1
        #plt.savefig('C:/Users/rmnin/Downloads/cwt images/' + str(i) + '.png')

for i in range(5000,5520):    
    #row = SmallDataSegment[i]
    #if np.prod(np.array(row)) == 0:
        #indices.append(i)
   
    #else:
    cwtmatr, freqs = pywt.cwt(row, np.arange(1, 552, 10), 'morl', method = 'fft' )
        #print(cwtmatr.shape)
    Y.append(np.array(cwtmatr))
    #img = plt.imshow(abs(cwtmatr), extent = [0, 20000, 1, 20000], cmap='gray', aspect='auto')
        #i += 1
  #plt.savefig('C:/Users/rmnin/Downloads/cwt images/' + str(i) + '.png')

print(Y)
Y = np.stack(Y)

# or
#Y = np.stack(Y, axis=0)

# or
#result_arr = np.vstack(result_arr)

array_2 = np.array(Y)
print(array_2)
print(array_2.shape)


# In[7]:




#Randomly choose 70% of index
number_of_rows = labeled.shape[0] #labeled is the variable we stored our labeled data in
seventypercentindices = np.random.choice(number_of_rows, size=3864, replace=False) #randomly chooses 70% of the indices in labeled

print(seventypercentindices)
print(seventypercentindices.shape)
#Apply random choice to both labels and matrices

training_matrix = array_2[seventypercentindices] #apply the randomly chosen indices to the 5520x56x240 matrix
training_labels = labeled[seventypercentindices] #apply the randomly chosen indices to the labeled data

#print(array_2[1099::])
#print(training_matrix)
#print(training_labels)
print(training_labels.shape)
print(training_matrix.shape)

#I'm not sure if it will cause any problems, so I wanted to let you know that the randomly chosen indices are not in numerical order. But I didn't think that would matter since we're applying the same indices to both the labels and the matrix. However, I wanted to let you know just in case.

#Get the remaining 30% for test data
thirtyperindices=[]
for val in range(labeled.shape[0]):
    if val not in seventypercentindices:
        thirtyperindices.append(val)
#for val not in seventypercentindices:
    #thirtyperindices.append(val)
#else:
   # thirtyperindices.append(i)
print(labeled.shape[0])
thirtypercentindices = np.array(thirtyperindices)
print(len(thirtyperindices))
print(len(seventypercentindices))
print(thirtypercentindices)

testing_matrix=array_2[thirtypercentindices]
testing_labels=labeled[thirtypercentindices]
print(testing_matrix.shape)
print(testing_labels.shape)




# In[10]:


# baseline cnn model for mnist
from numpy import mean
import numpy as np
import pandas as pd
from numpy import std
from matplotlib import pyplot
from sklearn.model_selection import KFold
import keras
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD
from keras.layers import BatchNormalization
from pandas import read_csv
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import random

# load train and test dataset
def load_dataset():
# load dataset
    (trainX, trainY), (testX, testY) = (training_matrix, training_labels), (testing_matrix, testing_labels)
    trainX= np.array([training_matrix], order='C')
#print(trainX)
#ranspose
# reshape dataset to have a single channel
    trainX = trainX.reshape((trainX.shape[0],240, 56, 1))
    testX = testX.reshape((240, 56, 1))
# one hot encode target values
    trainY = to_categorical(training_labels)
    testY = to_categorical(testing_labels)
return trainX, trainY, testX, testY

 
# scale pixels
def prep_pixels(train, test):
# convert from integers to floats
    train_norm = train.astype('float64')
    test_norm = test.astype('float64')
# normalize to range 0-1
    train_norm = train_norm / 255.0
    test_norm = test_norm / 255.0
# return normalized images
return train_norm, test_norm
 
# define cnn model
def define_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dropout(0.2, input_shape=(28,28)))
    model.add(Dense(10, activation='softmax'))
# compile model
    opt = SGD(lr=0.01, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
return model
 
# evaluate a model using k-fold cross-validation
def evaluate_model(dataX, dataY, n_folds=5):
    scores, histories = list(), list()
# prepare cross validation
    kfold = KFold(n_folds, shuffle=True, random_state=1)
# enumerate splits
    for train_ix, test_ix in kfold.split(dataX):
# define model
        model = define_model()
# select rows for train and test
        trainX, trainY, testX, testY = dataX[train_ix], dataY[train_ix], dataX[test_ix], dataY[test_ix]
# fit model
        history = model.fit(trainX, trainY, epochs=10, batch_size=32, validation_data=(testX, testY), verbose=0)
# evaluate model
        _, acc = model.evaluate(testX, testY, verbose=0)
        print('> %.3f' % (acc * 100.0))
# stores scores
        scores.append(acc)
        histories.append(history)
    return scores, histories
 
# plot diagnostic learning curves
def summarize_diagnostics(histories):
    for i in range(len(histories)):
# plot loss
        pyplot.subplot(2, 1, 1)
        pyplot.title('Cross Entropy Loss')
        pyplot.plot(histories[i].history['loss'], color='blue', label='train')
        pyplot.plot(histories[i].history['val_loss'], color='orange', label='test')
# plot accuracy
        pyplot.subplot(2, 1, 2)
        pyplot.title('Classification Accuracy')
        pyplot.plot(histories[i].history['accuracy'], color='blue', label='train')
        pyplot.plot(histories[i].history['val_accuracy'], color='orange', label='test')
        pyplot.show()
 
#summarize model performance
def summarize_performance(scores):
# print summary
    print('Accuracy: mean=%.3f std=%.3f, n=%d' % (mean(scores)*100, std(scores)*100, len(scores)))
# box and whisker plots of results
    pyplot.boxplot(scores)
    pyplot.show()
 
# run the test harness for evaluating a model
def run_test_harness():
# load dataset
    trainX, trainY, testX, testY = load_dataset()
# prepare pixel data
    trainX, testX = prep_pixels(trainX, testX)
# evaluate model
    scores, histories = evaluate_model(trainX, trainY)
# learning curves
    summarize_diagnostics(histories)
# summarize estimated performance
    summarize_performance(scores)
 
# entry point, run the test harness
run_test_harness()


# In[ ]:




