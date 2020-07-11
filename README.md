# Detection-and-Authentication-Banknote

Introduction
------------
Predicting authentication of banknote to determine if the banknote is Fake or Real using Artificial Neural Network ANN model.   
I am providing a documentation to show how I built and created the python algorithms and tools to make the prediction from both Artificial Neural Network to predict and classify if a banknote is real or fake. The model can be employed by financial istitution to curb, detect and autheticate banknotes The workflow is shown below. 

Getting Data
------------

To start with, the dataset used in the workflow (banknote_authentication_data) was accessed and downloaded from UCI Machine Learning Resiporatory. It contains 1372 banknotes with five attributes namely the banknote variances, skewness, curtosis, entropy and class. The dataset is named banknote_authentication_data.csv 

## Import data to project:

import numpy as np

import pandas as pd

import tensorflow as tf

## Importing the dataset

dataset = pd.read_csv('banknote_authentication_data.csv')

X = dataset.iloc[:, :-1].values

y = dataset.iloc[:, -1].values

This dataset was splited into "X", the independent variables containing all the varibles in the dataset with the exception of the "class" which is the dependent varriable "y".


Data Wrangling
--------------
After importing the dataset into google colab, the next step is to split the data into two namely; the Training set (X_train, y_train) and Test set (X_test and y_test). Following this, the independent variable data, X_train and X_test were Featured scaled separately using the StandardScaler library.

## Splitting the dataset into the Training set and Test set 

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


## Featured Scaling the independent the Training set and Test set 

#Feature Scaling 
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

Training the Dataset
--------------
After the dataset has been separated into the Training and Test set, I used the Artificial Neural Network (ANN) Model to build the ANN model and to Train the model on the Training dataset. 

##Initializing the ANN

ann = tf.keras.models.Sequential()

##Adding the input layer and the first hidden layer

ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

##Adding the second hidden layer

ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

##Adding the output layer

ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

## Training the ANN

##Compiling the ANN

ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

##Training the ANN on the Training set

ann.fit(X_train, y_train, batch_size = 32, epochs = 100)


## Predicting the results of the Test set from ANN Model 
The result from the ANN Model of the predicted y (y_pred) values from test set was compared to the observed y for the test set (y_test) values is shown below.

#Predicting the Test set results 

y_pred = ann.predict(X_test)
y_pred = (y_pred > 0.5)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

#Making the Confusion Matrix

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)

[[157   0]
 [  0 118]]

Accuracy: 1.000

Conclusions
--------------
In conclusion, the predictive results of the Authetication of Banknotes from the Artificial Neural Network model show a great accuracy indicating that autheticity of a banknote to reveal if it's real or fake can be accurately prdicted if using this model. This is a valuable model for the financial institutions.

