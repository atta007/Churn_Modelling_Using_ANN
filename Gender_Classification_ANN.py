# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 20:40:07 2018

@author: Atta
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.contrib.learn.python.learn.datasets.mnist import extract_images, extract_labels
from keras.utils import to_categorical
import ssl
import cv2
from keras.layers import Conv2D, Dense, Flatten, Dropout
from keras.models import Sequential

from scipy.io import loadmat

ssl._create_default_https_context = ssl._create_unverified_context
from keras.utils import np_utils


df = pd.read_csv('https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/wiki_crop.tar')


ROWS = 3#width
COLS =6 #height
CHANNELS =2 #rgb or grayscale
NUM_OF_CLASSES = 2# To classify data in number of 3 sets 
NUM_OF_SAMPLES =10000 # Images required for male and female.

# Retrieve and CLean Dataset
data = loadmat(df)

paths = df['wiki']['full_path'][0][0][0][:] #path to file
labels = df['wiki']['gender'][0][0][0][:]   #Images labels
labels = np.asarray(labels, dtype='int')

# Cleaning
garbage = list() #initialzing garbage List to clean garbage Values
for i,val in enumerate(labels):      #Iteriating through labels 
  if val == -9223372036854775808:    #Cleaning
      garbage.append(i)              #Adding garbage values to garbage[] list   
paths = np.delete(paths,garbage, axis=0) #Deleting garbage images from paths variables
labels = np.delete(labels,garbage)       #Deleting garbage images from labels varibales

#Selecting corrent Data
female = list() #initialzing female list
male = list()   #initialzing male list
for i,val in enumerate(labels):      #interating through data
  if val == 0 and len(female) != NUM_OF_SAMPLES:    #finding females labels through famle label value is == 0
    female.append(i)
                                                 
  if val == 1 and len(male) != NUM_OF_SAMPLES:      #finding Males labels through famle label value is == 1    
    male.append(i)                              #and 1 for male, NaN if unknown

to_select = male + female

paths = paths[to_select]    #X features
labels = labels[to_select]  # Y labels 

labels = np_utils.to_categorical(labels,NUM_OF_CLASSES) #label encoding 



#Initializing array for images
samples =10000 #total number of pics
X = np.ndarray(shape = (samples,ROWS,COLS,CHANNELS))


# Populating array of images to path to model plus normalizing it.
for i,image in enumerate(paths):
  
  image_path = paths+image[0]
  img = cv2.imread(image_path,1)
  img = cv2.resize(img,(ROWS,COLS),cv2.INTER_CUBIC) 
  img = img/255
  X[i] = img


 #Visualizing Data
plt.imshow(X[30])
plt.axis('off')
plt.show

# create your very own personel model
model = Sequential()


#Choose loss, optimizer and metrics
model.compile(loss ='binary_crossentropy' ,
             optimizer = 'adam',
             metrics =['accuracy'])


model.fit(#Features,
          #lables, 
          epochs = '2000',
          validation_split=labels)



from sklearn.model_selection import train_test_split
scores = model.evaluate(paths_test, labels_test)
print("%.2f"% scores[1])










