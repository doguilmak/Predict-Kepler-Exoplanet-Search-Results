# -*- coding: utf-8 -*-
"""
Created on Sat Aug 14 12:38:59 2021

@author: doguilmak

dataset: https://www.kaggle.com/nasa/kepler-exoplanet-search-results

"""
#%%
# 1. Importing Libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import time
import warnings
warnings.filterwarnings("ignore")

#%%
# 2. Data Preprocessing

# 2.1. Importing Data
start = time.time()
df = pd.read_csv('cumulative.csv')
print(df.head())

# 2.2. Remove Unnecessary Columns
df.drop(['koi_pdisposition','rowid','kepid','kepoi_name','kepler_name','koi_teq_err1','koi_teq_err2','koi_tce_delivname'], axis = 1, inplace = True)

"""
# 2.3. Creating correlation matrix heat map
plt.figure(figsize = (40,40))
sns.heatmap(df.corr(), annot=True)
"""

# 2.3. Looking for NaN Values and Replacing Mean Values of the Column
print("Number of NaN values: ", df.isnull().sum().sum())
df.fillna(df.mean(), inplace=True)
print("Number of NaN values: ",df.isnull().sum().sum())

# 2.4. Looking for Duplicated Datas
print("{} duplicated rows.".format(df.duplicated().sum()))

# 2.5. Looking For Anomalies
print(df.describe().T)

# 2.6. Determination of Dependent and Independent Variables
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df["koi_disposition"] = le.fit_transform(df['koi_disposition'])
y = df["koi_disposition"]
X = df.drop("koi_disposition", axis = 1)

#%%
# 3. Artifical Neural Network

# 3.1. Importing Libraries
from keras.models import Sequential
from keras.layers import Dense, Activation

# 3.2. Creating Layers
classifier = Sequential()
classifier.add(Dense(32, input_dim=41))
classifier.add(Activation('relu'))
classifier.add(Dense(64))
classifier.add(Activation('relu'))
classifier.add(Dense(32))
classifier.add(Activation('relu'))
classifier.add(Dense(8))
classifier.add(Activation('softmax'))

# 3.3. Compile and Fit the Data
classifier.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model_history = classifier.fit(X, y, epochs=128, batch_size=32, validation_split=0.13)

# 3.4. Plot accuracy and val_accuracy
print(model_history.history.keys())
plt.figure(figsize=(12, 12))
sns.set_style('whitegrid')
plt.plot(model_history.history['accuracy'])
plt.plot(model_history.history['val_accuracy'])
plt.title('ANN Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

import statistics
mean_val_accuracy = statistics.mean(model_history.history['val_accuracy'])
print(f'\nMean of validation accuracy: {mean_val_accuracy}') 

# 3.5 Predicting koi_disposition
predict = np.array([1, 0, 0, 0,	0, 9.48803557, 2.78E-05, -2.78E-05, 170.53875, 2.16E-03, -2.16E-03,	0.146, 0.318, -0.146, 2.9575, 0.0819, -0.0819, 6.16E+02, 1.95E+01, -1.95E+01, 2.26, 2.60E-01, -1.50E-01, 793, 93.59, 29.45,	-16.65,	35.8, 1, 5455, 81,-81, 4.467, 0.064, -0.096, 0.927,	0.105, -0.061, 291.93423, 48.141651, 15.347]).reshape(1, 41)  # 1st row of dataset.
if classifier.predict_classes(predict) == 0:
    print('\nModel predicted as FALSE POSITIVE.')
    print(f'Model predicted class as {classifier.predict_classes(predict)}.')
elif classifier.predict_classes(predict) == 1:
    print('\nModel predicted as CONFIRMED.')
    print(f'Model predicted class as {classifier.predict_classes(predict)}.')   
else:
    print('\nModel predicted as CANDIDATE')
    print(f'Model predicted class as {classifier.predict_classes(predict)}.')
    
end = time.time()
cal_time = end - start
print("\nProcess took {} seconds.".format(cal_time))
