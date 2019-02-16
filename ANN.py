# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 21:56:47 2018

@author: Atta
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values  
y = dataset.iloc[:, 13].values

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_country = LabelEncoder()
X[:, 1] = labelencoder_X_country.fit_transform(X[:, 1])

labelencoder_X_gender = LabelEncoder()
X[:, 2] = labelencoder_X_gender.fit_transform(X[:, 2])


onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

# Encoding the Dependent Variable
#labelencoder_y = LabelEncoder()
#y = labelencoder_y.fit_transform(y)





# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# Part 2 make ANN
import keras
from keras.models import Sequential
from keras.layers import Dense

#Initialising the ANN 
classifier=Sequential()

#Adding 1st Input layer and Hidden Layer 

classifier.add(Dense(output_dim = 6, init='uniform', activation='relu', input_dim=11))

#Addint 2nd Input layer and hidden layer
classifier.add(Dense(output_dim = 6, init='uniform', activation='relu'))

# Add the Output layer
classifier.add(Dense(output_dim = 1, init='uniform', activation='sigmoid'))

#Adding OutPut Layer
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics = ['accuracy'])

#Fitting the ANN to the training Set
classifier.fit(X_train, y_train, batch_size=10, nb_epoch=100)


# Fitting classifier to the Training set
# Create your classifier here

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)



from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense

def build_classifier():
    #Initialising the ANN 
    classifier=Sequential()
    
    #Adding 1st Input layer and Hidden Layer 
    
    classifier.add(Dense(output_dim = 6, init='uniform', activation='relu', input_dim=11))
    
    #Addint 2nd Input layer and hidden layer
    classifier.add(Dense(output_dim = 6, init='uniform', activation='relu'))
    
    # Add the Output layer
    classifier.add(Dense(output_dim = 1, init='uniform', activation='sigmoid'))
    
    #Adding OutPut Layer
    
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics = ['accuracy'])
    return classifier()

classifier = KerasClassifier(build_fn= build_classifier, batch_size=10, nb_epoch=100)
accuracies = cross_val_score(estimator= classifier, X= X_train, y=y_train, cv=10, n_jobs=-1)
mean = accuracies.mean()
variance = accuracies.std()